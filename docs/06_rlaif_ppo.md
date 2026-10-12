# 06 PPO 强化学习(RLAIF,`trainer/train_ppo.py`,447 行)

## 这一步在干什么(人话)

前面 SFT 教的是"照着标准答案说话",但模型从来没过自己"写完一篇答案,再听一句好不好"。这一步就让它**自己生成答案**,然后请一个只会打分的模型(奖励模型)给每条答案一个分数:分数好的写法以后多说,分数差的少说。

问题是模型很会"钻空子"——只要某种啰嗦写法能骗到高分,它就会一路刷分,把原来的说话能力说崩。所以这里加了两个刹车:一个是**别改动太猛**(clip + KL 惩罚),一个是**先自己估个预期分,只奖励"超出预期"的部分**(Critic 与优势)。

脚本实现标准 Actor-Critic PPO:采样生成 → 奖励模型打分(RLAIF:用 AI 反馈代替人工标注)→ Critic 逐 token 估值 → GAE 算优势 → clip 目标更新 Actor/Critic。产物 `out/ppo_actor_768.pth`(`--save_weight ppo_actor` `train_ppo.py:309`,保存 `:284-291`)。

## 先懂这几个词

| 术语 | 大白话 |
|---|---|
| 强化学习(RL) | 没有标准答案,靠"打分"试错:做了被表扬的事就多做了 |
| 策略(policy) | 模型当前的"说话习惯",即给定上文如何逐个选下一个词 |
| Actor / 策略模型 | 被训练的那个说话模型本体,就是 SFT 后的权重 |
| Rollout / 采样 | 让模型自己把一道题答完(`generate`),留下可打分的答卷 |
| 奖励 reward | 一条答案的分数(这里是奖励模型给的分 + 几条规则分) |
| 奖励模型(Reward Model, RM) | 只会打分的另一个模型,不改答案也不被训练;类比阅卷老师 |
| RLAIF | 用 AI 打分代替人工打分,省掉标注的人(Reinforcement Learning from AI Feedback) |
| 价值函数 V(s) / Critic | 模型自己"估计我这句大概能得几分";Critic 是专门估分的第二个模型 |
| 优势(advantage) | 实际得分 − 预期得分,即"超出我自己预期的那部分"才值得被强化 |
| GAE | 把这份"意外之喜"沿 token 从后往前逐位摊开的算法(λ 控制摊多远) |
| EOS | 结束符,标记"我说完了";奖励只挂在这个位置 |
| 重要性采样 ratio | 新习惯 ÷ 采样时旧习惯对同一个词的概率比,衡量"这次改动有多大" |
| clip | 限速器:改动超过阈值就不再给梯度,防止一步跨太远 |
| KL 散度 | 两个说话习惯差多远;这里用来"拉住别跑偏" |
| minibatch / PPO epoch | 同一批采到的答卷重复拿来更新几次(`--ppo_update_iters`) |
| 早停(early stop) | 检测到已经跑太远(KL 超阈值)就提前结束本轮重复更新 |

## 五个模型角色(初始化在 `train_ppo.py:376-388`,引擎 `:390-399`)

为什么要五份?因为一次 RL 训练里同时有五种活:出题、打分、估预期分、拉住别跑偏、把题答完。它们不能共用一个模型。

| 角色 | 来源(代码) | 作用 |
|---|---|---|
| Actor(策略) | `init_model(from_weight='full_sft')`(`:379`) | 生成答案 + 唯一被真正更新、最终交付的模型 |
| Critic | `CriticModel`(`:36-48`):继承 `MiniMindForCausalLM`,加 `value_head = nn.Linear(hidden_size, 1)`(`:40`),`forward`(`:42-48`)输出逐 token 标量 V(s);权重同样从 full_sft 载入(`:385-386`,`strict=False` 容忍新头) | 逐 token 估计 V(s),给优势当基线 |
| Reference | `init_model(...)` + `eval().requires_grad_(False)`(`:380-381`) | full_sft 的冻结副本,提供 KL 惩罚参照 |
| Reward Model | `LMForRewardModel`(`trainer_utils.py:162-181`,InternLM2-1.8B-Reward,默认路径 `../../internlm2-1_8b-reward` `train_ppo.py:337`,即与 minimind 仓库同级的目录;打分接口 `get_score` `trainer_utils.py:170`) | 对整条回复打分 |
| Rollout Engine | `create_rollout_engine`(`rollout_engine.py:209`) | 采样生成;默认 `--rollout_engine torch`(`:346`) |

Rollout 引擎不算"第六份模型":Torch 引擎只是持有 Actor 的引用调 `generate`(`rollout_engine.py:71-92`),`update_policy` 就是换个引用(`:94-95`);SGLang 引擎是外部服务进程,靠落盘 + `/update_weights_from_disk` 换权重(`:175-194`)。

数据:`rlaif.jsonl`(`RLAIFDataset`,`lm_dataset.py:199`)。它**只提供 prompt,`answer` 是空字符串**(`lm_dataset.py:225-228`)——这正是 RL 和 SFT 的区别:没有标准答案可用,只有题目。prompt 由 `apply_chat_template` 拼出(`:212-220`),并按 `--thinking_ratio 0.9`(`train_ppo.py:345`)的概率带上"先思考再回答"的模板标记(`lm_dataset.py:214`)。

默认超参(`argparse` 区 `train_ppo.py:307-350`):bs=2(`:311`)、actor lr=3e-7(`:312`)、critic lr=5e-7(`:313`)、max_prompt=768(`--max_seq_len` `:324`)/ max_gen=1024(`--max_gen_len` `:325`)、`clip_epsilon=0.2`(`:327`)、`vf_coef=0.5`(`:328`)、`kl_coef=0.02`(`:329`)、`gamma=1.0`(`:330`)、`lam=0.95`(`:331`)、`cliprange_value=0.2`(`:332`)、`ppo_update_iters=2`(`:333`)、`early_stop_kl=0.25`(`:334`)、`mini_batch_size=2`(`:335`)。

运行(路径全是相对路径,必须先 `cd trainer`):

```bash
cd trainer && python train_ppo.py
```

## 一步的流程(`ppo_train_epoch`,`train_ppo.py:78-303`)

### 1. Rollout 采样(`:85-99`)——解决"没有答案可学,先自己造答案"

`tokenizer(prompts, padding_side="left", max_length=args.max_seq_len)`(`:85-86`)后调 `rollout_engine.rollout(num_generations=1, max_new_tokens=args.max_gen_len, temperature=0.8)`(`:88-94`),拿回生成 token、`completion_ids`、`prompt_lens`、文本 `responses_text` 和采样时的 `old_resp_logp`(`:95-99`)。PPO 每个 prompt 只采 **1 条**(`num_generations=1` `:91`),对比 GRPO 的一组见 [07](07_grpo_cispo.md)。

### 2. 计算奖励(`:100` + `calculate_rewards` `:51-75`)——解决"这条答案到底几分"

分数由几部分累加(不是只看奖励模型):

- 长度门限:`20 <= len(response.strip()) <= 800` 给 +0.5,否则 −0.5(`:61`)——防"太短敷衍"和"堆字数"
- thinking 格式:有 `</think>` 时,思考段 20~300 字给 +1.0 否则 −0.5(`:64`),`</think>` 恰好出现 1 次给 +0.25 否则 −0.25(`:65`)
- 重复惩罚:`rewards -= rep_penalty(answer)`(`:67`),`rep_penalty`(`:29-32`)按 3-gram 重复率扣分,上限 `cap=0.5`
- 奖励模型分:`reward_model.get_score(messages, answer)`(`:69`),整体加进 rewards(`:72-73`);分数被夹在 ±3(`trainer_utils.py:179`)。messages 是从 chat 模板里正则还原出的 role/content(`:57-59`)

**奖励只有序列级一个标量**,再放到"说完"那一格:先按首个 EOS 截断出有效长度(`:124-126`),`token_rewards` 全零、只在 `last_idx = resp_lengths - 1` 处加外部奖励(`:136-138`)。这就是"稀疏奖励":中间每个字本身没有分数。

### 3. Critic 估值(`:130-133`)——解决"光有绝对分数不够,得知道好不好于预期"

`no_grad` 下用 Critic 跑整条序列,`gather` 出 completion 位置的 `old_resp_values` 并乘 `resp_value_mask`(`:131-133`);同一块 `no_grad` 里还算出 Reference 的 `ref_resp_logp`(`:135`),供 KL 用。

### 4. GAE(`:140-147`)——解决"末尾一个分数,怎么分给前面每个 token"

```python
for t in reversed(range(gen_len)):                                              # :141
    nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0                  # :142 末位之后不 bootstrap
    delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]        # :143
    lastgaelam = delta + args.gamma * args.lam * lastgaelam                      # :144
advantages = torch.stack(advs_rev[::-1], dim=1)                                  # :146
returns = advantages + old_resp_values                                           # :147
```

即 `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)`,`A_t = δ_t + γλ·A_{t+1}`,`returns = A + V`。默认 `gamma=1.0`(`:330`)→ 逐 token 不打折;`lam=0.95`(`:331`)→ 意外之喜主要摊给附近 token。之后按 mask 做 batch 内均值/方差标准化(`:149-151`),优势再乘 `resp_policy_mask`。

### 5. 多轮 minibatch 更新(`:162-254`)——解决"一批样本太贵,榨干它但别榨过头"

`mb_size = max(1, min(args.mini_batch_size, B))`(`:153`),外层 `for ppo_epoch in range(args.ppo_update_iters)`(`:162`)、随机打乱后切块(`:165-167`)。每个 minibatch:

- Critic 前向出新 value(`:169-170`),Actor 在 `autocast_ctx` 前向、`log_softmax` gather 出新 per-token logp(`:172-175`)
- `log_ratio = mb_resp_logp − old_resp_logp`(`:177`)、`ratio = exp(log_ratio)`(`:201`)
- `approx_kl = 0.5·log_ratio²` 的 mask 均值(`:191`)——**只用于早停判断和日志**,不进 loss
- 早停:`approx_kl` 先跨卡 `all_reduce(AVG)`(`:194-196`)再和 `early_stop_kl` 比较(`:198-199`)
- `clipfrac`:`|ratio − 1| > clip_epsilon` 的 token 占比(`:202-203`),纯监控指标
- KL 惩罚(进 loss,与上面那个 approx_kl 是两回事):`exp(ref_logp − policy_logp) − (ref_logp − policy_logp) − 1` 的 mask 均值(`:204-205`),乘 `kl_coef`(`:209`)
- **Actor clip 目标**(`:206-208`):`max(−A·ratio, −A·clamp(ratio, 1±ε))` 的 mask 均值,等价于最大化 `min(A·ratio, A·clamp(ratio))`(取 max 是因为这里是 loss,要最小化)
- **Critic clip-VF 目标**(`:210-213`):`0.5·max((V_new − returns)², (clamp(V_new, V_old ± cliprange_value) − returns)²)` 的 mask 均值,`vf_coef` 加权进总 loss(`:222`)
- 总 loss `= (policy_loss + vf_coef·value_loss + aux_loss) / accumulation_steps`(`:222`);`aux_loss` 只在 MoE 时非零(`:174`)
- actor / critic 两套优化器分别梯度裁剪再 step(各自 scheduler / zero_grad,`:236-244`);本轮结束时若还有残留梯度,再做一次收尾更新(`:246-254`)

### 6. 同步推理权重(`:256`)——解决"生成用的模型和训练的模型不是一个"

每 `save_interval` 步(或最后一步)`rollout_engine.update_policy(actor_model)`。Torch 引擎直接替换引用(`rollout_engine.py:94-95`);SGLang 引擎把新权重落盘再请求 `/update_weights_from_disk`(`:175-194`),失败会 `raise`(`:193`)。

### 7. 存权重(`:284-297`)——解决"存什么、存几份"

主进程把 Actor 转 fp16 存 `out/ppo_actor_768.pth`(`:285-291`);`lm_checkpoint(...)` 额外把 critic / critic_optimizer / critic_scheduler 一起存进 `checkpoints/`(`:294-297`),供 `--from_resume 1` 恢复(`:360`、`:412-420`)。**Critic 不进 `out/`,不能拿它去推理。**

## 新手常见坑

1. **必须在 `trainer/` 下运行**:数据、权重、奖励模型路径全是相对路径(`--data_path ../dataset/rlaif.jsonl` `:326`、`--save_dir ../out` `:308`、`--reward_model_path ../../internlm2-1_8b-reward` `:337`)。在仓库根目录跑,`../..` 就指到别处,RM 加载直接失败。
2. **别把 `out/` 里的 `ppo_actor_768.pth` 当整个训练成果**:只有 Actor 被存成推理权重(`:285-291`),Critic 只在 `checkpoints/` 的 resume 文件里(`:294-297`)。Actor 与 Critic 各有独立 lr 和优化器(`:402-403`、`:239-240`),改 lr 要改对参数(`--learning_rate` `:312` / `--critic_learning_rate` `:313`)。
3. **两个 KL 不是一个东西**:`approx_kl`(`:191`,新 vs 采样旧策略,只触发早停和日志)和 `kl_ref_penalty`(`:204-205`,策略 vs Reference,乘 `kl_coef` 真进 loss `:209`)。想"拉得更紧"要调 `kl_coef`,调 `early_stop_kl` 只会让更新次数变少。
4. **早停不会中断前向/反向**:命中后 loss 乘 `0.0`(`:219-220`)而不是 `break`,是为了让 DDP 的通信闭环不断(`:193-196` 的注释)。看到 loss 突然为 0、日志里的数还在动,是正常的,不是 bug。

## 下一步

PPO 能用,但"为了估一个预期分,要额外训练并显存常驻一个 Critic"代价不小。[07 GRPO/CISPO](07_grpo_cispo.md) 讲怎么干脆把 Critic 去掉:同一个问题采一组答案,用组内平均分当基线。
