# 07 GRPO / CISPO(`trainer/train_grpo.py`,326 行)

## 这一步在干什么(人话)

GRPO(Group Relative Policy Optimization,DeepSeekMath 提出)要做的事和 PPO 一样:让模型自己写答案、按分数调整倾向。区别在**怎么知道"这个分数算高还是低"**。

PPO 的办法是额外训练一个 Critic(第二个模型)来估"预期分",代价是多养一份模型、多调一套超参。GRPO 的办法简单得多:**同一个问题连采 6 条答案,拿这 6 条的平均分当预期分**——比平均分高的强化,比平均分低的削弱。于是 Critic 整个被删掉。

CISPO(MiniMax-M1 提出)在本脚本里作为 `--loss_type cispo` 分支,而且是**默认值**(`train_grpo.py:221`)。它改的是"梯度怎么给":原版 GRPO 会把改动过大的 token 直接 clip 到零梯度,CISPO 只把权重截一个上限、梯度全部保留。产物 `out/grpo_768.pth`(`--save_weight grpo` `:202`,保存 `:180-187`)。

## 先懂这几个词

| 术语 | 大白话 |
|---|---|
| 组(group)/ `num_generations` | 同一道题连采几条答案,默认 6(`:219`);类比同一件事试 6 次,再比较哪次好 |
| baseline / 基线 | "算不算好"的参照点;GRPO 用组内平均分当参照,不用 Critic |
| 优势 advantage | 这条答案的分数比组内平均分高多少(还除以组内标准差) |
| rollout / 采样 | 让模型把答案生成完;`old logprob` 是采样那一刻记录的"我当时说这个词有多确定" |
| 重要性采样 ratio | 现在说这个词的概率 ÷ 采样时的概率,衡量更新跑偏了多少 |
| clip | 限速器:ratio 越界就不再给这个 token 梯度 |
| detach | 把某个数当常数用,梯度不从它流过去 |
| per-token logprob | 模型对每个词的自信程度的对数;越大表示越"想这么说" |
| KL 散度 | 当前说话习惯和冻结的 Reference 差多远,防止刷分刷到跑偏 |
| 奖励模型 / RLAIF | 只会打分的另一个模型(AI 反馈),省掉人工标注 |
| EOS | 结束符;loss 只算到第一个 EOS 为止 |
| mask | 哪些 token 该计入 loss 的 0/1 标记 |
| 左 padding | 把 prompt 的补齐空格加在**左边**,让"下一个要生成的词"紧贴右端 |

## 关键超参(`argparse` 区 `train_grpo.py:201-236`)

| 参数 | 行 | 默认 | 含义 |
|---|---|---|---|
| `--num_generations` | `:219` | 6 | 每 prompt 采样条数(组大小) |
| `--beta` | `:220` | 0.1 | KL 惩罚系数(对 Reference) |
| `--loss_type` | `:221` | **cispo** | 可选 `grpo` / `cispo`,决定走哪个分支 |
| `--epsilon` | `:222` | 0.2 | 只在 **grpo 分支**用作两侧 clip 半径(`:139`);cispo 分支不读它 |
| `--epsilon_high` | `:223` | 5.0 | cispo 分支唯一的截断:ratio 的上界(`:136`) |
| `--batch_size` | `:204` | 2 | 每 step 的 prompt 数,× 6 后实际 12 条回复 |
| `--learning_rate` | `:205` | 3e-7 | 只有一份优化器(`:287`) |
| `--max_seq_len` / `--max_gen_len` | `:216` / `:217` | 768 / 1024 | prompt 与生成长度 |
| `--data_path` | `:218` | `../dataset/rlaif.jsonl` | 同 PPO,只有 prompt、`answer` 为空(`dataset/lm_dataset.py:225-228`) |
| `--thinking_ratio` | `:232` | 0.9 | prompt 里带上"先思考"模板的概率(`dataset/lm_dataset.py:214`) |
| `--rollout_engine` | `:233` | torch | 采样引擎(`trainer/rollout_engine.py:209`) |

运行(路径全是相对路径,必须先 `cd trainer`):

```bash
cd trainer
python train_grpo.py                    # 默认 cispo
python train_grpo.py --loss_type grpo   # 原版 GRPO clip 目标
```

## 一步的流程(`grpo_train_epoch`,`train_grpo.py:71-196`)

### 1. 组采样(`:74-94`)——解决"没有标准答案,先让模型自己答出好几条"

prompt 用 **左 padding** 编码:`padding_side="left"`(`:74-75`),并保留**尾部** `max_seq_len` 个 token(`:76-78`)——生成要从句子右端续写,补齐加在左边才不会把答案和 prompt 隔开。`rollout_engine.rollout(num_generations=args.num_generations, max_new_tokens=args.max_gen_len, temperature=0.8)`(`:80-86`)内部靠 `repeat_interleave` 把每个 prompt 复制 G 份(`rollout_engine.py:76-77`),于是 batch 从 `B` 变成 `B*num_gen`。

`old_per_token_logps` 直接取采样时算出的 per-token logp(`:90`),由引擎用同一个待更新模型算(`rollout_engine.py:88` → `compute_per_token_logps` `:24-36`)——这就是 ratio 的分母,所以刚采完第一步 ratio≈1。

### 2. 奖励(`:96` + `calculate_rewards` `:37-68`)——解决"每条答案各给一个分数"

和 PPO 同一套规则:长度 20~800 字 +0.5/−0.5(`:54`)、thinking 段长度与 `</think>` 次数加分(`:55-58`)、3-gram 重复惩罚 `rep_penalty`(`:60`、`:31-34`)、再叠奖励模型分(`:62-63`、`:65-66`;分数夹在 ±3,`trainer_utils.py:179`)。双层循环按 `i*num_generations + j` 索引(`:44-46`),保证组内顺序和 `repeat_interleave` 一致。输出形状 `[B*num_gen]`。

### 3. 当前策略与 Reference 的 logprob(`:98-104`)——解决"要拿现在和过去、和原始模型比"

`res.logits[:, :-1, :]` 做 `log_softmax` 后 gather 出 completion 位置每个 token 的 logp(`:101`),`ref_model` 在 `no_grad` 下同样算一遍(`:103-104`;Reference 是 full_sft 的冻结副本 `:269-270`)。

### 4. 组内相对优势(`:121-124`)——解决"分数绝对值没意义,要的是比同组其它答案好多少"

```python
grouped_rewards = rewards.view(-1, args.num_generations)                    # :121 [B, G]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations) # :122 组均值
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # :123 组标准差
advantages = (rewards - mean_r) / (std_r + 1e-4)                             # :124
```

这就是 GRPO 的"免 Critic baseline":每条答案的优势 = 自己的分减组均值、除组标准差,它是一个**整条序列一个标量**,再靠 `advantages.unsqueeze(1)` 广播到每个 token(`:137`、`:140-141`)——所以 GRPO 没有逐 token 的 credit assignment,这与 PPO 的 GAE 是本质差别。

### 5. EOS 截断 mask(`:126-130`)——解决"模型一旦说完就不该再为后面负责"

找到每条回复的首个 EOS(`:127-129`),mask 取 `<= eos_idx`(EOS 本身计入)并与 padding mask 相与(`:130`)。EOS 之后的内容是 padding,不参与 loss。

### 6. KL 估计(`:132-133`)——解决"别为了刷分把原有能力说崩"

```python
kl_div = ref_per_token_logps - per_token_logps            # :132 Δ
per_token_kl = torch.exp(kl_div) - kl_div - 1             # :133
```

`e^Δ − Δ − 1` 是 KL 的无偏低方差估计式(DeepSeek 做法),它逐 token 非负,比直接拿 `Δ` 当惩罚稳。

### 7. 策略 loss:两个分支(`:134-142`)——解决"这个分数到底该怎么变成梯度"

`ratio = exp(per_token_logps − old_per_token_logps)`(`:134`)。

**CISPO 分支(`:135-137`,默认)**

```python
clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()          # :136
per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)  # :137
```

特征:`.detach()`(`:136`)让 ratio 只当**常数权重**,梯度全部走 `per_token_logps`;截断只有上界 `epsilon_high`(`:223`,默认 5.0)、**没有下界、也没有 `min` 式的置零**。

**GRPO 分支(`:138-142`)**

```python
clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)       # :139
per_token_loss1 = ratio * advantages.unsqueeze(1)                            # :140
per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)                    # :141
per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)  # :142
```

即 PPO 式双 clip(两侧都夹、取 `min` 保守版),且**没有 detach**,梯度可以从 ratio 本身流过去。

**为什么要从 GRPO 换成 CISPO**(对应上面两处代码差别,不是文档口号):正优势的好 token 一旦被 `clamp(…, 1+ε)`(`:139`)夹住,`min` 之后梯度归零——一个已经说得挺顺、还想让它说得更好的 token 反而学不动。CISPO 把权重截上限但保留 `per_token_logps` 的梯度(`:136-137`),token 级 loss 变成"重加权的 MLE 形式",所有 token 都还在学,只是学多少的问题。

### 8. 平均与更新(`:143-152`)——解决"长答案不该因为字多就压过短答案"

```python
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()  # :143
loss = (policy_loss + aux_loss) / args.accumulation_steps                                                       # :144
```

先按每条回复的**有效 token 求均值**(`:143`,除数 clamp 防零),再对**序列求均值**;`aux_loss` 只在 MoE 时非零(`:100`)。反向在 `:145`;到累积点或最后一步才做梯度裁剪 + step(`:147-152`,`--grad_clip` `:210`)。

### 9. 存权重与同步采样权重(`:180-193`)

主进程 fp16 存 `out/grpo_768.pth`(`:180-187`),`lm_checkpoint` 存可续训状态(`:188-189`);每 `save_interval` 步把最新权重交回采样引擎(`:193`)。日志里 `Reward / KL_ref / Adv Std / Adv Mean / Actor Loss / Avg Response Len`(`:164-167`)。

## 与 PPO 对比

| | PPO(`train_ppo.py`) | GRPO / CISPO(`train_grpo.py`) |
|---|---|---|
| Critic | 需要:`CriticModel`(`:36-48`)、独立优化器与 lr(`:403`、`:313`) | 不需要(无 value 头,只有 `:287` 一个优化器) |
| 优势来源 | GAE:`train_ppo.py:141-147`(逐 token) | 组内标准化:`train_grpo.py:121-124`(整条一个标量) |
| 每 prompt 采样数 | 1(`:91`) | G=`num_generations`,默认 6(`:219`) |
| 加载的语言模型份数 | 3 份(actor `:379` + ref `:380` + critic `:385-387`) | 2 份(policy `:267` + ref `:269`) |
| 奖励模型 | 1 份,fp16(`:388`) | 1 份,fp16(`:272`) |
| KL 权重 | `kl_coef=0.02`(`:329`),k3 式估计(`:204-205`) | `beta=0.1`(`:220`),k3 式估计(`:132-133`) |
| clip | 两侧 `clip_epsilon=0.2`(`:327`) | grpo 两侧 `epsilon=0.2`(`:222`)/ cispo 仅上界 `epsilon_high=5.0`(`:223`) |
| 重复利用同批数据 | `ppo_update_iters=2` 轮 minibatch(`:162`、`:333`) | 每批只过一遍(`:98-145` 在 step 循环内) |
| 采样显存压力 | 每 step 2 条(`:311`) | 每 step 2×6=12 条(`:204` × `:219`),单步序列数直接×6 |

## 新手常见坑

1. **默认跑的是 CISPO,不是 GRPO**:`--loss_type` 默认 `cispo`(`:221`)。想复现原版 GRPO 的 clip 目标必须显式 `--loss_type grpo`,否则会疑惑"为什么 `--epsilon` 调了没反应"——cispo 分支只读 `epsilon_high`(`:136`),`epsilon` 只在 `:139` 被用。
2. **日志里的 `KL_ref` 和 loss 里的 KL 不是同一个式子**:loss 用 `e^Δ − Δ − 1`(`:133`),日志用 `Δ` 的均值(`:159`)。它可以是负数,别当成 KL 恒正,也别拿它和 `beta` 直接对算。
3. **组内全同分则这条样本没法学**:`advantages = (rewards − mean_r)/(std_r + 1e-4)`(`:124`),一组 6 条分数一样时 `std_r=0`、分子为 0,优势归零。温度 0.8(`:85`)和 `num_generations`(`:219`)就是为了让组内拉开差距。
4. **别在仓库根目录直接跑**:`--data_path ../dataset/rlaif.jsonl`(`:218`)、`--save_dir ../out`(`:201`)、`--reward_model_path ../../internlm2-1_8b-reward`(`:225`)全是相对路径,必须先 `cd trainer`;奖励模型用 `trust_remote_code` 加载(`trainer_utils.py:164-165`),路径不对就报加载失败而不是精度问题。

## 下一步

到这里"用 AI 打分调模型倾向"就通了。接着看 [08 Agentic RL](08_agentic_rl.md):把打分从"像不像人话"换成"工具调用有没有真的算对";采样与引擎细节见 [10 Rollout 引擎](10_rollout_engine.md),跑完想验证效果见 [11 评估/部署](11_eval_deploy.md)。
