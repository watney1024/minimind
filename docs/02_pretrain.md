# 02 预训练 Pretrain(`trainer/train_pretrain.py`,163 行)

## 这一步在干什么(人话)

这一步先把一个随机参数的大脑搭起来,然后让它埋头读大量纯文本(百科、新闻、问答等),自己总结语言的统计规律。它唯一的练习方式是:看着前半句,猜下一个词。猜错就罚分,罚分就是 loss(损失,一个表示错得有多离谱的数字,越小越好)。

因为不需要人工标注标准答案,这一步常被称为无监督训练,也是整个项目最花钱、最花时间的一步。跑完你会得到 `out/pretrain_768.pth`,它很会续写你给它的话,但还不会对话:你问它问题,它可能接着往下编问题,而不是回答你。要让它学会回答,靠下一步的 SFT([03 全参 SFT](03_full_sft.md))。

模型规模:默认 `--hidden_size 768` 加 `--num_hidden_layers 8`(`train_pretrain.py:88-89`)的 64M 小模型,词表 6400 个 token(`model/model_minimind.py:18`)。

## 先懂这几个词

| 术语 | 大白话 |
|---|---|
| token | 模型眼里的字。中文大约 1 个 token 对应 1.5~1.7 个汉字(见 `train_pretrain.py:90` 的参数说明),英文约 0.75 个单词 |
| 分词 / tokenizer | 把文字切成 token 编号的那本字典,本项目词表 6400 |
| next-token prediction | 猜下一个 token。语言模型的训练目标就是这一件事 |
| loss(损失) | 这一批数据上模型平均错多少,越小越好,是训练时唯一要盯的曲线 |
| 交叉熵 / CE loss | 一种衡量模型给正确答案分配的概率够不够高的 loss 算法,本项目直接用它(`model/model_minimind.py:257`) |
| 困惑度 ppl | loss 的指数换算 `e^loss`,可理解成模型平均在多少个候选词里犹豫。loss 2.5 约等于 ppl 12 |
| 梯度 | 每个参数该往哪个方向调一点点的指示箭头 |
| 优化器 / AdamW | 拿着梯度真正去改参数的模块,AdamW 是常用的一种(`train_pretrain.py:130`) |
| 学习率 lr | 每次改参数迈多大步。太大会训崩,太小太慢,本步默认 5e-4 |
| batch(批) | 一次喂给模型的样本条数,默认 32(`train_pretrain.py:79`) |
| step / iter | 一次前向加一次反向。代码里的 `iters` 就是这个 epoch 一共多少个 step |
| epoch(轮) | 把整份训练数据完整过一遍叫 1 个 epoch,默认跑 2 轮 |
| gradient accumulation(梯度累积) | 显存装不下大 batch 时,先跑几个 step 把梯度攒起来再一起更新。默认攒 8 个,等效 batch = 32×8 = 256 |
| gradient clipping(梯度裁剪) | 梯度箭头太长就按比例截短,防止一步迈太猛把模型练坏。阈值 `--grad_clip` 默认 1.0(`train_pretrain.py:85`) |
| 混合精度 / bf16 | 用 16 位小数做中间计算来省显存提速,而不是 32 位。默认 `--dtype bfloat16`(`train_pretrain.py:82`) |
| DDP | 多张显卡各算各的 batch、再把梯度互相同步后一起更新。用 `torchrun` 启动就自动开启 |
| checkpoint / 断点续训 | 把模型参数、优化器状态、训到第几步存盘,中断后从那一步接着练而不是从头再来 |
| warmup / decay | 学习率先由小变大叫热身,再由大变小叫收尾。本脚本只有收尾衰减,没有热身 |
| MoE / aux loss | MoE 是把每层拆成多个专家、每个 token 只激活其中几个的结构;aux loss 是防止所有 token 都挤去同一个专家的额外罚分。默认 `--use_moe 0`,此时恒为 0 |

## 超参默认值(逐项核对 `train_pretrain.py:75-99`)

下表是不写任何命令行参数直接跑时生效的值,逐项对照脚本里的 argparse 定义核对过。

| 参数 | 默认 | 行号 | 大白话说明 |
|---|---|---|---|
| `--epochs` | 2 | `:78` | 整份数据过 2 遍 |
| `--batch_size` | 32 | `:79` | 单卡一次喂 32 条样本 |
| `--learning_rate` | 5e-4 | `:80` | 初始学习率,随后按余弦曲线降到它的 10% |
| `--accumulation_steps` | 8 | `:84` | 攒 8 个 step 的梯度再更新一次,等效 batch 256 |
| `--max_seq_len` | 340 | `:90` | 每条样本固定切到 340 个 token,超出的部分直接截断 |
| `--data_path` | `../dataset/pretrain_t2t_mini.jsonl` | `:93` | 训练语料,每行一个 JSON,只有一个 `text` 字段 |
| `--from_weight` | `none` | `:94` | 不基于任何已有权重,即随机初始化从零开始 |
| `--save_interval` | 1000 | `:87` | 每 1000 个 step 存一次权重和续训状态 |
| `--hidden_size` / `--num_hidden_layers` | 768 / 8 | `:88-89` | 模型宽度与层数,决定 64M 这个体量 |
| `--use_moe` | 0 | `:91` | 关掉 MoE,此时 aux_loss 恒为 0 |
| `--dtype` | bfloat16 | `:82` | 混合精度类型 |
| `--grad_clip` | 1.0 | `:85` | 梯度裁剪阈值 |
| `--log_interval` | 100 | `:86` | 每 100 step 打一行日志 |
| `--num_workers` | 8 | `:83` | 数据加载线程数 |
| `--seed` | 42 | `:92` | 随机种子,DDP 下每个 rank 为 seed+rank,每轮为 seed+epoch |
| `--save_dir` / `--save_weight` | `../out` / `pretrain` | `:76-77` | 输出目录与权重名前缀 |
| `--from_resume` | 0 | `:95` | 置 1 才自动检测断点续训 |
| `--use_compile` | 0 | `:98` | 置 1 启用 torch.compile 加速 |

关于 `--max_seq_len`:340 是脚本默认值(`:90`)。因为每条样本都会被补齐到这个固定长度(见下面第 3 个坑),它直接决定显存占用和每步耗时。旧文档里主线 380 / mini 768 的推荐值在本次核对中没有找到代码或 README 依据,已删去。

## 主流程(`train_pretrain.py:101-164`)

main 函数里本来就有 9 段带 `========== N. ... ==========` 注释的代码块,按代码里真实出现的顺序如下(分布式、续训、混合精度这些公共件的细节见 [docs/README.md](README.md) 公共训练基建一节):

1. **初始化环境和随机种子**(`:101-104`):`init_distributed_mode()` 判断是不是 `torchrun` 启动(`:102`),是则把 `args.device` 换成该 rank 对应的显卡(`:103`),再按 rank 设随机种子(`:104`)。
2. **配置目录、模型参数、检查 ckp**(`:106-109`):建输出目录(`:107`)、造 `MiniMindConfig`(`:108`)、仅当 `--from_resume 1` 时去读 `checkpoints/pretrain_768_resume.pth`(`:109`)。
3. **设置混合精度**(`:111-114`):决定 bf16 还是 fp16,并准备好 `autocast_ctx`。
4. **配 wandb**(`:116-123`):只有传了 `--use_wandb` 且是 0 号进程才生效,实际导入的是 `swanlab`(`:119`)。
5. **定义模型、数据、优化器**(`:125-130`):`init_model` 建模型和 tokenizer(`:126`)、`PretrainDataset`(`:127`)、DDP 采样器(`:128`)、`GradScaler`(`:129`)、AdamW(`:130`)。
6. **从 ckp 恢复状态**(`:132-139`):第 2 步读到的续训数据在这里灌回 model / optimizer / scaler / epoch / step。
7. **编译和分布式包装**(`:141-146`):`torch.compile`(`:142-143`)、DDP 包装(`:145-146`)。
8. **开始训练**(`:148-159`):逐 epoch 重排样本(`:150-151`),续训时用 `SkipBatchSampler` 跳过已训 step(`:152-157`),否则正常训练(`:159`)。
9. **清理分布进程**(`:161-164`):`dist.barrier()` 等所有卡训完,再 `destroy_process_group()`。

> 旧版本文档在这里漏了第 4 步(配 wandb)和第 9 步(清理进程),编号因此从 3 跳到了 5。

## 训练目标到底是什么:猜下一个字

这是全篇最该看懂的一段。

`PretrainDataset` 把一条文本切成 token 后,前面补一个起始符、后面补一个结束符,再补填充符凑满固定长度(`dataset/lm_dataset.py:52-55`);然后直接把这个序列原样复制一份当答案(`dataset/lm_dataset.py:56-57`)。也就是说:**答案就是原文自己,只是整体向右移一位**。填充符的位置被标成 `-100`,而 `-100` 是交叉熵里约定的忽略标记(`model/model_minimind.py:257`),所以补出来的空白不参与打分。

模型内部做的移位就一行(`model/model_minimind.py:256`):

```python
x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
```

意思是:第 i 个位置的输出,负责预测第 i+1 个 token。所以训练时模型看到的每个前缀都在变长,而它每一步只被要求做同一件事——猜下一个 token。这就是 next-token prediction。

### loss 从 9 掉到 2 意味着什么

交叉熵可以直接换算成困惑度 `ppl = e^loss`,它约等于模型在猜每个 token 时平均在多少个候选里犹豫:

| loss | ppl = e^loss | 人话 |
|---|---|---|
| 8.8 | 6400 | 完全随机:等于在 6400 个词表里瞎蒙,一点信息都没用上 |
| 4 | 55 | 学到了一些词频和局部搭配,但仍在几十个候选里犹豫 |
| 2.5 | 12 | 大部分常见字能猜准,遇到人名、数字、少见搭配还会犹豫 |
| 2 | 7 | 只有七八个合理候选,语言基本功已经到位 |

loss 的起点有硬依据:6400 词表下随机初始化的交叉熵约等于 `ln(6400) ≈ 8.8`。仓库公开的 768 维预训练曲线(`README.md:687` 引用的 `images/pretrain_loss.jpg`)也确实是 9 以下起步、约 1.5 收尾;注意那张图是项目方的发布曲线,不是你这次跑 mini 数据一定会得到的数值。

两个必须知道的边界:

- **loss 低不等于知识多、逻辑对**,只说明它很擅长预测下一个 token。
- **loss 高不一定笨**,也可能是数据里有大量噪声、乱码或不可预测内容。

## 单步训练(`train_epoch`,`train_pretrain.py:24-71`)

这一步解决的问题是:怎么把一批发进去的样本变成一次参数更新。

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):   # :26 续训时 step 从断点之后接着数
    input_ids = input_ids.to(args.device)                                    # :27 数据搬上显卡
    labels = labels.to(args.device)                                          # :28

    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)  # :29 算出本步学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr                                               # :31 手动写回优化器

    with autocast_ctx:
        res = model(input_ids, labels=labels)                                # :34 前向,模型内部算 CE loss
        loss = res.loss + res.aux_loss                                       # :35 加上 MoE 负载均衡辅助 loss
        loss = loss / args.accumulation_steps                                # :36 累积窗口内取平均

    scaler.scale(loss).backward()                                            # :38 反向传播,算出梯度
```

每攒够 `accumulation_steps` 个 step 才真正更新一次(`:40-47`):

```python
if step % args.accumulation_steps == 0 or step == iters:                     # :40 攒够 8 个,或已到本轮末尾
    scaler.unscale_(optimizer)                                               # :41 还原梯度,否则裁剪会裁错
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)       # :42 梯度裁剪
    scaler.step(optimizer)                                                   # :44 更新参数
    scaler.update()                                                          # :45
    optimizer.zero_grad(set_to_none=True)                                    # :47 清空梯度,准备下一窗口
```

三个容易看错的点:

- 更新参数的是 `scaler.step(optimizer)`(`:44`),不是直接 `optimizer.step()`。GradScaler 包了这一层,才能在 fp16 下自动跳过溢出步。
- 默认 dtype 是 bfloat16,而 `GradScaler(enabled=(args.dtype == 'float16'))`(`:129`)只在 fp16 时才启用。也就是说**默认配置下 scaler 只是个透传壳子**,留着是为了你换 fp16 时不用改代码。
- 最后一个 step 用 `or step == iters`(`:40`)兜底,保证不满 8 个的零头也会更新一次,不会白训。

## 学习率:只有收尾衰减,没有热身

`get_lr`(`trainer/trainer_utils.py:42-43`)的真实公式:

```python
return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))
```

代入两端:第 0 步系数是 `0.1 + 0.45×2 = 1.0`,即满速 5e-4;走到最后一步系数是 0.1,即降到 10%(5e-5)。中间是余弦曲线下降,不是直线。

注意两点:第一,`current_step` 传的是 `epoch * iters + step`(`train_pretrain.py:29`),所以衰减是按**全程总步数**算的,不会每个 epoch 重来一遍;第二,`iters` 来自 `len(loader)`,多卡下 `DistributedSampler` 会把数据切给各张卡,所以卡越多每轮 step 数越少,曲线在绝对步数上会整体变短(但形状仍是覆盖全程)。

> 旧版文档把这里写成线性 warmdown 并给出 `lr * (1 - step/total)`,与代码不符:代码是余弦下降,且有 10% 下限,不会降到 0。

## 日志字段(`train_pretrain.py:49-57`)

解决的问题是:一行日志要能同时看出主 loss、MoE 辅助项、学习率和剩余时间。

| 字段 | 来源 | 含义 |
|---|---|---|
| `loss` | `:51` | 本步总 loss,即主 loss 加 aux loss。注意它由 `loss.item() * accumulation_steps` 还原,抵消了 `:36` 的除法 |
| `logits_loss` | `:53` | 纯语言建模 loss,等于 `loss - aux_loss`,判断模型学语言学得好不好就看它 |
| `aux_loss` | `:52` | MoE 负载均衡辅助 loss;`--use_moe 0` 时恒为 0(`model/model_minimind.py:180`、`:236`) |
| `lr` | `:54` | 当前学习率 |
| `epoch_time` | `:55-56` | 名字有误导:它是**本轮剩余分钟数**的估计,不是已用时间。算法是已用时间除以已跑 step 再乘剩余 step |

`logits_loss` 与 `aux_loss` 的拆分只对 MoE 有意义。默认 64M 非 MoE 跑起来,`loss` 和 `logits_loss` 数值相同。

## 存权重(`train_pretrain.py:59-69`)

解决的问题是:一次保存同时要产出推理能用的文件和续训能用的文件。

- 推理文件:`out/pretrain_768.pth`,文件名由 `{save_weight}_{hidden_size}{moe后缀}.pth` 拼出(`:61-62`)。存盘前会把参数转成 fp16 并搬回 CPU(`:66`),所以这个文件比训练时的内存占用小一半。
- 续训文件:`checkpoints/pretrain_768_resume.pth`,由 `lm_checkpoint()`(`:67`)写入,里面除了模型还有 optimizer、scaler、epoch、step(`trainer/trainer_utils.py:87-106`)。
- 换卡数也能续:恢复时如果发现保存时的 GPU 数和现在不一样,会按比例换算 step(`trainer/trainer_utils.py:112-116`)。
- 只有 0 号进程存盘(`:59` 的 `is_main_process()`),避免多卡互相覆盖。
- 每次保存是覆盖同名文件,不是留快照,所以中途不会堆出几十个权重文件。

## 一次训练量有多大(可算,不是实测)

按当前仓库里的数据文件行数除以默认 batch size,可以直接算出每轮 step 数:

- `dataset/pretrain_t2t_mini.jsonl` 共 1,270,238 行,`batch_size=32` → 单卡每轮 39,695 个 step(公式就是 `SkipBatchSampler.__len__`,`trainer/trainer_utils.py:157-159`)。
- 多卡时 `DistributedSampler` 会把这些样本切给各张卡,每卡 step 数按卡数等比减少。
- 这个数会随你换数据文件而变,不是固定值。
- 单步的 token 量是 `batch_size × max_seq_len = 32 × 340 = 10,880`,和 SFT 默认配置的 `16 × 768 = 12,288` 在同一量级(见 [03 全参 SFT](03_full_sft.md))。

速度和显存本次拿不到代码或仓库层面的依据,所以不给数字。两点说明:

- 旧文档同时写了 ~4 step/s 和 ~50min/epoch,这两个数互相矛盾(39,695 ÷ 4 ≈ 2.8 小时,不是 50 分钟),已删。
- 仓库里唯一有记录的耗时是根 README 的一句:2 小时指 **SFT** 阶段单张 3090 跑完 1 个 epoch 的实测耗时(`README.md:42`),不是预训练。按上面单步 token 量同量级、SFT 每轮 step 数更多来推,预训练 1 个 epoch 的量级不会比它更夸张;这只是按代码默认值做的推算,不是实测,也不是保证值。

## 运行

```bash
cd trainer
python train_pretrain.py                        # 单卡
torchrun --nproc_per_node 4 train_pretrain.py   # 4 卡 DDP(根 README.md:680 的写法是 nproc_per_node 1)
python train_pretrain.py --from_resume 1        # 中断后自动续训
```

训完可以先冒烟测一下,确认它确实学会了续写:

```bash
cd .. && python eval_llm.py --weight pretrain
```

## 新手常见坑

1. **以为 loss 会掉到 0**。不会。语言本身有不可预测性,而且本脚本的学习率有 10% 下限(`trainer/trainer_utils.py:43`),后期曲线就是在某个值附近抖动。看到 1.5~2 附近平坦是正常的,不是训练坏了。
2. **把 `epoch_time` 读成已用时间**。它是剩余分钟数估计(`train_pretrain.py:55-56`),而且 `start_time` 在每次进入 `train_epoch` 时重置(`:25`),所以它只反映本轮剩余时间。
3. **默认 `max_seq_len=340` 会静默截断长文本**。`PretrainDataset` 只保留前 `max_length-2` 个 token(`dataset/lm_dataset.py:52`),超出部分直接丢掉且不报警。同时每条样本都被补齐到这个固定长度(`:54`),所以显存和每步耗时由它决定,与原文实际长短无关。
4. **`strict=False` 会静默吞掉名字对不上的层**。`init_model` 加载权重用的是 `model.load_state_dict(weights, strict=False)`(`trainer/trainer_utils.py:129`),checkpoint 里没有的键会被直接跳过、不报错,那部分参数就停留在随机初始化状态。注意边界:这只对**键名**不生效,一旦**形状**对不上仍会抛 RuntimeError(本次实测确认)。所以改了 `--hidden_size` / `--num_hidden_layers` / `--use_moe` 之后,要盯一下启动时打印的 `Model Params` 和 `Trainable Params`(`trainer_utils.py:30-31`、`:132`),别以为不报错就是加载成功了。反过来,`--from_weight` 写成一个不存在的权重名则是另一种表现:直接 `FileNotFoundError`,因为路径是按 `out/{from_weight}_{hidden_size}{moe后缀}.pth` 拼的(`:126-127`),它不会退回随机初始化。

## 下一步

预训练只造出一个会续写的底座。要让它变成能对话的助手,得用带角色标记的对话数据、并且只在助手回复上算 loss:[03 全参 SFT](03_full_sft.md)。
