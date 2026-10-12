# 05 DPO 偏好对齐(`trainer/train_dpo.py`,218 行)

## 这一步在干什么(人话)

到 [03 全参 SFT](03_full_sft.md) / [04 LoRA](04_lora.md) 之后,模型已经会正常回答问题了,但交叉熵只教它"照着一份标准答案接龙",没教过它**同一问题的两种回答里哪种更好**——而这种"更好"往往没有唯一标准答案,只有相对偏好。

传统 RLHF 要走两段:先拿偏好数据训一个"打分员"(奖励模型),再用强化学习(PPO,见 [06 PPO/RLAIF](06_rlaif_ppo.md))让模型去讨好打分员。DPO(Direct Preference Optimization)的关键洞察是这两段能合成一段:把"chosen 优于 rejected"这件事直接写成一个**可求导的分类 loss**,于是既不用训奖励模型,也不用真的去采样生成。

所以本篇的代码非常薄:一条数据管线(每个样本给同一问题的两个版本答案)+ 一个 17 行的 loss + 两份前向(正在训练的模型自己,加一份冻结的"参考模型")。产出是完整的 `out/dpo_768.pth`。

## 先懂这几个词

| 术语 | 大白话 |
|---|---|
| token | 分词器切出来的最小文本单位,模型每步只预测下一个 token(见 [00 分词器与数据](00_tokenizer_dataset.md)) |
| 偏好对 | 同一个问题配上两个回答:更符合偏好的叫 chosen(记作 w,winning),差一点的叫 rejected(记作 l,losing) |
| 策略模型(π / policy) | 正在被训练的那个模型;它每读一个 token 就对词表里每个词打一个分,分数变成概率,概率决定下一步说什么 |
| 参考模型(ref) | 训练开始时把同一份权重**再复制一份**,全程冻结,只用来提供"原来的口味"作为基准 |
| logits | 模型输出的原始分数向量(词表那么大一个),还没归一化成概率 |
| softmax | 把一堆分数按大小压成加起来等于 1 的概率分布 |
| log 概率 | 概率取自然对数。概率在 0~1 之间,取 log 后是负数,越接近 1 越接近 0 |
| 序列 log 概率 | 一句回答里每个 token 的 log 概率加起来(= 整句出现概率取 log),越长越负 |
| loss | 一个标量,衡量"现在错得多厉害",训练就是让它变小 |
| σ / logsigmoid | σ(x)=1/(1+e^{-x}),把任意数压到 0~1 的"置信度";`-log σ(x)` 就是"希望 x 越大越好"的损失 |
| β | 刻度盘:控制允许模型偏离参考模型多远 |

## 数据:`DPODataset`(`dataset/lm_dataset.py:125-196`)

要解决的问题:一条偏好样本到底长什么样,以及"问题(prompt)"这部分藏在哪里。

数据文件默认 `../dataset/dpo.jsonl`(`trainer/train_dpo.py:140`),需要自己下载(README.md:459 起,格式说明在 README.md:465-476;本仓库 `dataset/` 目录里不带这个文件)。每行 JSON 的形状是 `chosen` 和 `rejected` **各自都是一整段对话**(含 user 的那一问),而不是单独一个 answer 字段。

- `__getitem__`(`lm_dataset.py:138-178`):取 `sample['chosen']` / `sample['rejected']`(`:140-141`),各自完整套一次 chat template(`:142-148`),然后 truncate + pad 到 `max_length`(`:152-157`)。
  - 也就是说**张量里没有独立的 prompt 字段**:两个版本各自是"整条对话"的 token 序列,共用的 user 提问也编在序列里,靠 mask 把它排除掉。
  - `:149-151` 有个细节:是否删掉空的 `<think>...</think>` 标签由一次随机决定,chosen 和 rejected **共用同一个结果**(`remove_empty_think` 只掷一次;移除逻辑在 `lm_dataset.py:31-38`),避免两个版本因为格式处理不同而被不公平地比较。
- 输入输出错位(`:164-169`):`x = ids[:-1]`、`y = ids[1:]`、`mask = loss_mask[1:]`,让模型看前 t 个 token 猜第 t+1 个,mask 与 y 严格对齐。
- `generate_loss_mask()`(`:180-196`):扫描助手回复段的起止标记(起止 id 在 `:131-132` 由 `<|...|>assistant\n` 和 `eos\n` tokenize 得到),**只把助手回复那几段置 1**,user 部分和 padding 都是 0。这就是"只对模型写出来的那部分计损失"的机制。
- 一个关键约定:**batch 的前一半放 chosen,后一半放 rejected**。
  - 组装侧:`train_dpo.py:57-65` 把两者按这个顺序 `torch.cat([x_chosen, x_rejected], dim=0)`(y、mask 同样顺序 cat),DataLoader 不做任何打乱。
  - 切分侧:`dpo_loss`(`:41-44`)按 `[:batch_size // 2]` 当 chosen、`[batch_size // 2:]` 当 rejected。
  - 两侧必须同时成立:chosen/rejected 的身份**只由行号位置决定**,没有任何标签字段。好处是两者共用同一次前向(拼成一个 batch),省一半算力。
- 成本注意:每条样本都 pad 到 `max_length`(`:152-157`,`--max_seq_len` 默认 1024,`train_dpo.py:138`),所以显存和耗时基本按最长序列算,不会因为这条回答很短就变便宜。

## 核心 loss(`trainer/train_dpo.py:25-50`)

要解决的问题:怎么把"A 比 B 好"这种相对判断,变成一个能对参数求导的数。

### 第一步:把 logits 换成"这句回答有多大概率"

`logits_to_log_probs`(`:25-31`)两行搞定(各步 shape 见 `:26-28` 的注释):

```python
log_probs = F.log_softmax(logits, dim=2)                                  # :29 每个位置、每个 token 的 log 概率
log_probs_per_token = torch.gather(log_probs, 2, labels.unsqueeze(2))     # :30 只挑"真实出现的那个 token"那一列
```

`(batch, seq_len, vocab_size)` → `(batch, seq_len)`:位置 t 上模型给出的分布里,实际出现的 `y[t]` 拿到的那条 log 概率。

### 第二步:`dpo_loss`(`:34-50`)

```python
ref_log_probs = (ref_log_probs * mask).sum(dim=1)        # :36 只留助手回复段,按句求和 → 序列 log 概率
policy_log_probs = (policy_log_probs * mask).sum(dim=1)  # :37
chosen / reject 各切一半                                  # :41-44
pi_logratios  = chosen_policy - reject_policy            # :46 新模型对"好−坏"的区分度
ref_logratios = chosen_ref    - reject_ref               # :47 参考模型对"好−坏"的区分度
logits        = pi_logratios - ref_logratios             # :48 区分度的"净提升"
loss          = -F.logsigmoid(beta * logits)             # :49
return loss.mean()                                       # :50
```

用人话把 `:49` 这个式子翻一遍(等价于 README.md:953 的公式
`-log σ(β[log(π_w/π_ref_w) − log(π_l/π_ref_l)])`,`:46-49` 逐项对得上):

```
loss = -log σ( β · [ (新模型:好答案 − 坏答案) − (参考模型:好答案 − 坏答案) ] )
```

1. 里面每个括号是同一句回答在新/老模型下的序列 log 概率,相减得到"新模型比老模型更喜欢好答案多少"和"新模型比老模型更喜欢坏答案多少"。
2. `:46` `pi_logratios` = 新模型对好答案的偏好强度;`:47` `ref_logratios` = 训练前那个自己的偏好强度。
3. `:48` 相减 = **相对参考模型的净偏好提升**。这正是 DPO 的要点:它不要求模型"喜欢好答案"(它早就会了),只要求这件事**比训练前的自己更突出**。净提升为正 → σ 接近 1 → loss 接近 0;净提升为负 → loss 变大。
4. `:49` `β`(默认 0.15,`:143`)把这个净提升放大后再进 `-log σ`:β 越大,同样的优势被放大越多,梯度越猛、模型被允许离开参考模型越远;β 越小越保守。
5. `:36-37` 的求和是**不做长度归一**的原始和:chosen 和 rejected 长度差很多时,长度会直接混进这个比较里,代码里没有除长度这一项。

顺带一个可推出的性质:训练第一步时策略模型和参考模型权重完全相同(`:174` 与 `:177` 加载同一个 `--from_weight`),模型里唯一的随机性来源 dropout 默认是 0(`model/model_minimind.py:17`),所以同一批输入下 `:48` 的 `logits` 恰好为 0,`-log σ(0) = ln 2 ≈ 0.69`,这就是 loss 的起点。

## 训练流程(`trainer/train_dpo.py`)

要解决的问题:loss 里那两个"老模型的分"从哪来,以及到底谁在被更新。

- 策略模型:`init_model(lm_config, args.from_weight, ...)`(`:174`),默认 `--from_weight full_sft`(`:141`)。优化器吃**全部参数**:`optim.AdamW(model.parameters(), lr=args.learning_rate)`(`:185`),梯度裁剪也是整个模型(`:89`)。注意 DPO 是全参训练,和 04 的 LoRA 没有半点关系。
- 参考模型:再 `init_model` 一份同权重的副本(`:177`),`eval()` + `requires_grad_(False)`(`:178-179`),不进优化器、只负责前向。两者参数量分别打印在 `:175` 和 `:180`,即显存里同时驻着两份模型——README.md:961 的说法是"DPO 只需跑 actor 与 ref 两个模型,显存占用低、收敛稳定、实现简单",这正是它比 06 里 PPO 好落地的原因。
- `train_epoch`(`:53-119`)一个 step 的骨架:
  1. 取 chosen/rejected 并 cat(`:57-65`)
  2. `with torch.no_grad():` 跑参考模型前向拿 `ref_logits`(`:72-74`),转成 log 概率(`:75`)——冻结 + no_grad 双保险,保证 ref 永不更新
  3. 正常前向跑策略模型拿 `logits`(`:77-79`)。注意 `model(x)` **不传 labels**,所以模型自带的交叉熵 loss 没被用到,唯一的监督信号来自 `dpo_loss`
  4. `dpo_loss_val = dpo_loss(...)`(`:81`),`loss = dpo_loss_val + outputs.aux_loss`(`:82`),再除 `accumulation_steps`(`:83`)。`aux_loss` 是 MoE 路由的辅助项,稠密模型上为 0(`model/model_minimind.py:236`)
  5. `scaler.scale(loss).backward()`(`:85`)→ 攒够 `accumulation_steps` 才裁剪、`optimizer.step()`、`zero_grad`(`:87-92`)
  6. 日志把 `dpo_loss` 和 `aux_loss` 分开打印(`:97-102`),看对齐进展主要看 `dpo_loss`
- 学习率默认 `4e-8`(`:128`,参数注释写明"建议 <=5e-8 避免遗忘"),比全参 SFT 的 `1e-5`(`trainer/train_full_sft.py:81`)小三个数量级。原因很直白:模型已经会说话了,这一步只调"偏好口味",步子一大就会把语言能力本身训坏。
- 其他默认:`--epochs 1`(`:126`)、`--batch_size 4`(`:127`)、`--max_seq_len 1024`(`:138`)、`--beta 0.15`(`:143`)。`train_epoch` 形参里那个 `beta=0.1`(`:53`)只是默认值,实际调用总会把命令行参数传进去(`:212`、`:214`)。
- 保存(`:106-114`):`f'{save_dir}/{save_weight}_{hidden_size}{moe后缀}.pth'`(`:109`),默认 `--save_weight dpo`(`:125`)→ `out/dpo_768.pth`,全部权重转 fp16 落盘(`:113`)。产物是完整模型,评测时(在仓库根目录)`python eval_llm.py --weight dpo`(`--weight` 就是这个前缀名,`eval_llm.py:36`,拼路径见 `eval_llm.py:22`)。

## 运行

```bash
cd trainer && python train_dpo.py                # 数据默认 ../dataset/dpo.jsonl,需先下载到 dataset/
cd trainer && torchrun --nproc_per_node 2 train_dpo.py    # 多卡(README.md:966-969 给的两种方式)
```

`--data_path ../dataset/dpo.jsonl`(`:140`)、`--save_dir ../out`(`:124`)、`../checkpoints`(`:157`)都是**相对当前目录**的路径,所以必须站在 `trainer/` 里跑,否则数据文件会找不到(多卡命令同理,README 根目录那份 `torchrun ... trainer/train_xxx.py` 需要你把 `--data_path`/`--save_dir` 换成 `./dataset/...`、`./out`)。

## 新手常见坑

- **cat 顺序是唯一的契约**:chosen/rejected 的身份只由"在前一半还是后一半"决定(`:63-65` 的拼接顺序 与 `:41-44` 的切分)。改了数据管线、加了新字段、想混入单答案样本,必须两侧同时改;否则 loss 会安静地把好坏对调,变成教模型喜欢差答案,程序一行报错都没有。
- **`--batch_size` 必须是偶数**:`:40-44` 靠 `batch_size // 2` 整除切分,传奇数会错位,chosen/rejected 配对直接乱掉。默认 4(`:127`)是安全的。
- **loss 不降 ≠ 训练失败,降了也 ≠ 模型变好**:`dpo_loss` 只衡量"相对参考模型的偏好净提升",起点固定在 ln 2 ≈ 0.69(见上),它完全不约束"模型整体输出概率掉多少"。如果 loss 在降但输出变差、变短或跑题,先怀疑 lr 太大(`:128`)或 β 太大(`:143`)。
- **参考模型必须与起点同源且不更新**:`:177` 用同一个 `--from_weight` 再加载一次、`:179` 冻结。为了省显存把 ref 换成别的权重(比如 pretrain),β 所代表的"别离初始自己太远"就失去意义了。

## 下一步

DPO 只能吃现成的偏好对、不会自己试错;想让模型在可自动判对错的任务上在线生成、在线拿奖励,继续看 [06 PPO(RLAIF)](06_rlaif_ppo.md)。
