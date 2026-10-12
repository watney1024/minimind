# 04 LoRA 微调(`trainer/train_lora.py`,176 行)

## 这一步在干什么(人话)

到 [03 全参 SFT](03_full_sft.md) 为止,你手里有一个会聊天的 `full_sft` 模型,但它"什么都只会一点":不知道自己的身份设定,也不懂医疗这类垂域的说法方式。想教它新东西,全参微调当然可以,可那要把 6400 万个参数全部重训一遍,还容易把原有的通用能力练丢。

LoRA 的做法是:**原来那些权重一个都不动**(冻结),只在每个想改的权重矩阵旁边**并联**一条很小的"旁路"——两个瘦长小矩阵相乘,近似一个"低秩的改动量"。训练只更新这条旁路,算输出时把旁路的输出**加**到原输出上。

省在哪:反向传播要为每个可训练参数存梯度和优化器状态(AdamW 还要额外存两份动量),需要更新的参数少了 99% 以上,这部分开销几乎就没了;产物也只有旁路那一小包数,而不是整个模型。

用的时候有两条路,别搞混:**叠加**(推理时基座 + 旁路一起加载,换任务只换小文件)和**合并**(把旁路算进原权重,得到一个不再需要旁路的完整模型)。

## 先懂这几个词

| 术语 | 大白话 |
|---|---|
| 微调 | 在已经训好的模型上,用新数据接着训一小会儿,让它偏向新任务 |
| 基座 | 已经训好、被你在下面垫着用的那个模型,这里指 03 产出的 `full_sft` 权重 |
| 冻结参数 | 设 `requires_grad=False`,告诉 PyTorch"这个矩阵不用算梯度、不用更新" |
| 线性层 | 就是"乘一个大矩阵(再加偏置)",是模型里参数量最集中的地方 |
| 秩 / 低秩 | 不直接改大矩阵,而是用"大×小"和"小×大"两个矩阵相乘来近似这次改动;中间那个"小"的宽度就叫 rank,越小越省 |
| 旁路(低秩分支) | 原矩阵旁边多出来的一条短路:输入同时走原路和这条路,两路结果相加 |
| 可训练参数 | `requires_grad=True` 的那部分参数,只有它们会被优化器更新 |
| loss | 一个标量,衡量"现在错得多厉害";训练就是让它变小 |
| 叠加 vs 合并 | 叠加=基座和旁路分开存、推理时一起加载;合并=把旁路的乘积加进基座权重,产物是完整模型 |

## LoRA 本体(`model/model_lora.py`,65 行)

要解决的问题:怎么在原矩阵旁边挂一条小旁路,而且**刚挂上的那一瞬间模型输出完全不变**——否则基座的能力会被随机初始化的小矩阵立刻打乱。

- `LoRA` 模块(`model/model_lora.py:6-18`):两个不带 bias 的线性层,`A: in_features → rank`(`:10`)、`B: rank → out_features`(`:11`),前向就一行 `self.B(self.A(x))`(`:17-18`)——先把输入压到 rank 维,再展回来。
  - 注意:这份手写实现**没有**论文里的 `alpha/rank` 缩放系数(`model/model_lora.py` 全文没有 alpha 这个量),旁路输出直接相加。
- 起点等价原模型靠的是初始化(`:13-15`):`A` 用均值 0、标准差 0.02 的高斯随机数,`B` **全部置零**。因为 `B` 是 0,`B(A(x)) ≡ 0`,挂上后输出 = 原输出 + 0,与原模型严格等价;反向传播时梯度先进 `B`、再进 `A`,`B` 才会慢慢离开 0。
- `apply_lora(model, rank=16)`(`:21-32`):
  - 挂谁的判据是"**方阵线性层**":`isinstance(module, nn.Linear) and module.in_features == module.out_features`(`:23`)。**不是按层名挑 q/k/v/o**。
  - 实测(默认 768 维 / 8 层):每层只有 `self_attn.q_proj` 和 `self_attn.o_proj` 被挂上,共 16 个模块。`k_proj/v_proj` 挂不上,因为它们是 GQA 压缩过的 768→384(`model/model_minimind.py:101-102`)不是方阵;MLP 的 `gate/up/down_proj`(768↔2432,`:140-142`)、`lm_head`(768→6400,`:246`)、MoE 路由 `gate`(`:152`)同理都挂不上。MoE 版配置下也是同样的 q_proj/o_proj 两类。
  - 挂法:`setattr(module, "lora", lora)` 把旁路注册成子模块(`:25`),再用闭包把 `module.forward` 换成 `layer1(x) + layer2(x)`(`:29-32`)。这是在**实例上打补丁**,不是往网络里加新层。
  - `apply_lora` **不负责冻结**,冻结在训练脚本里做(下一节)。
  - `rank` 只能通过函数参数改(`:21`);`train_lora.py` 调它时没传(`trainer/train_lora.py:121`),所以实际就是 16,命令行没有 `--rank` 选项。
- 存取旁路:`save_lora`(`:45-53`)遍历带 `lora` 属性的模块,只导出 `{层名}.lora.A.weight / .lora.B.weight`,顺手转成 fp16(`:51`);`load_lora`(`:35-42`)反过来按 `lora.` 前缀灌回每个旁路(前提是旁路已经被 `apply_lora` 挂好)。
- `merge_lora`(`:56-65`):先 `load_lora` 读旁路(`:57`),再对每个挂了旁路的方阵层做 `weight += B.weight @ A.weight`(`:64`),最后只保存不含 `.lora.` 的键(`:59`)。这就是"把旁路并进原矩阵"的样子,和训练时 `原forward(x) + B(A(x))` 数学上等价。

## 训练脚本要点(`trainer/train_lora.py`)

要解决的问题:怎么保证"只有旁路被更新,基座一动不许动"。主流程在 `train_lora.py:119-143`。

- 加载基座并挂旁路:`init_model(lm_config, args.from_weight, ...)`(`:120`)→ `apply_lora(model)`(`:121`)。默认 `--from_weight full_sft`(`:88`),即从 `out/full_sft_768.pth` 读(`trainer/trainer_utils.py:127`),所以这一步必须先跑完 03。
- 参数量打印(`:124-128`):分别数总参数、名字含 `lora` 的参数,再打印占比。实测:768 维稠密模型总 64.305M、LoRA 0.393M、**0.61%**;同尺寸 MoE 版 0.20%,hidden_size=512 时 0.87%。"~1%"不是通用数字,只是小模型上的接近值。
- 冻结 + 收集(`:130-137`):遍历 `named_parameters()`,名字含 `'lora'` 的设 `requires_grad=True` 并收进 `lora_params`,其余一律 `requires_grad=False`。
- 数据:直接复用 03 那套 `SFTDataset`(`:140`,类在 `dataset/lm_dataset.py:61-122`)。LoRA 不需要新数据格式,`{"conversations":[...]}` 的多轮对话 jsonl 即可;默认垂域数据 `../dataset/lora_medical.jsonl`(`:87`),换领域时改 `--data_path` 指向对应的 jsonl(README.md:777 的约定是"准备同样的多轮对话格式数据,放到 `lora_xxx.jsonl`")。
- **优化器只喂 `lora_params`**:`optim.AdamW(lora_params, lr=args.learning_rate)`(`:143`)。基座既在 `:136-137` 被冻结、又不在优化器的参数表里,双重保证不会被更新——省显存主要就是省在这里(不用为 6400 万个参数存梯度和 Adam 状态)。
- **梯度裁剪也只作用于 `lora_params`**:`torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)`(`:43`),阈值默认 1.0(`:80`)。
- loss 和全参 SFT 一模一样:`res.loss + res.aux_loss`(`:36`),还是"猜下一个 token"的交叉熵,LoRA 没有引入新目标函数。
- 保存(`:58-64`):路径 `f'{save_dir}/{lora_name}_{hidden_size}{moe后缀}.pth'`(`:61`),默认 `--lora_name lora_medical`(`:72`)→ `out/lora_medical_768.pth`;真正落盘的只有旁路(`save_lora`,`model/model_lora.py:45-53`,fp16);断点续训状态另走 `lm_checkpoint`(`:64`)。
- 超参默认值:lr `1e-4`(`:75`,比全参 SFT 的 `1e-5`(`trainer/train_full_sft.py:81`)大一个数量级,因为只有旁路在学)、batch_size 32(`:74`)、epochs 10(`:73`)、`max_seq_len 340`(`:85`,垂域数据短,序列截得短也是省算力的关键)。

前缀名由 **`--lora_name`**(`:72`)决定;注意它和推理侧的 `--lora_weight` 不是同一个参数名。

## 使用:训练 / 叠加 / 合并

要解决的问题:训完这一小包旁路权重之后,怎么让模型真的用上它。

```bash
# 训练(默认即医疗垂域:--lora_name lora_medical + ../dataset/lora_medical.jsonl)
cd trainer && python train_lora.py
# 换领域时,权重名和数据路径要一起改(权重名决定产物文件名)
cd trainer && python train_lora.py --lora_name lora_identity --data_path ../dataset/你的领域数据.jsonl

# 方式 A:推理时叠加(基座 + 旁路分别加载,旁路不落进基座;需在仓库根目录)
python eval_llm.py --weight full_sft --lora_weight lora_identity
```

`eval_llm.py` 这条命令和 README.md:797、README.md:802 给的示例一致。

- 注意路径都相对当前目录:`--data_path ../dataset/...`(`:87`)、`--save_dir ../out`(`:71`)、`../checkpoints`(`:64`),所以训练要站在 `trainer/` 里跑;而 `eval_llm.py` 读的是 `./out/...`(`eval_llm.py:22`),要站在仓库根目录跑。
- 叠加的实现路径:`eval_llm.py:24-26` 先 `apply_lora(model)` 再 `load_lora(model, './out/{lora_weight}_{hidden_size}{moe}.pth')`;API 服务一样(`scripts/serve_openai_api.py:41-43`)。
- 因此三个条件必须同时满足:推理侧模型结构与训练时一致(`--hidden_size`/`--use_moe`)、先 `apply_lora` 才有地方可灌、文件名对得上(训练给 `--lora_name lora_identity`,推理就得 `--lora_weight lora_identity`)。
- 叠加的意义:一个基座可以配多份旁路(身份、医疗各一份),按需替换那个很小的旁路文件(实测 768 维下只有 0.393M 个参数),也能随时退回原始基座。

```bash
# 方式 B:永久合并回基座,得到不依赖旁路的完整权重
cd scripts && python convert_model.py    # 需先把 convert_merge_base_lora 的调用取消注释
```

- 注意 `convert_model.py` 的 `__main__` 默认跑的是 torch→transformers 导出(`scripts/convert_model.py:128-134`),合并那几行是注释掉的(`:136-140`),要先取消注释再运行。
- 合并入口 `convert_merge_base_lora`(`scripts/convert_model.py:105-112`):加载基座 → `apply_lora` → `merge_lora` 落盘。
- 合并之后就是一般的完整模型权重,评测不再需要 `--lora_weight`;代价是你再也退不回"原始基座 + 可插拔"的状态,而且每份领域数据都要占一整份模型文件。

## 新手常见坑

- **以为 `apply_lora` 帮你冻结了**:它只挂旁路(`model/model_lora.py:21-32`),冻结是 `train_lora.py:130-137` 那段循环干的。自己照抄全参脚本、把 `model.parameters()` 喂给优化器(对照 `:143`),就会连基座一起训,却还以为在做 LoRA。
- **被加载阶段的打印误导**:`init_model` 在挂旁路**之前**就打印了一次 `Trainable Params`(`trainer/trainer_utils.py:132`),那还是 60 多 M;真正的旁路占比要看 `train_lora.py:124-128` 打出的那三行。
- **`--lora_name` 与 `--lora_weight` 搞混**:训练侧是 `--lora_name`(`train_lora.py:72`),推理侧是 `--lora_weight`(`eval_llm.py:37`、`scripts/serve_openai_api.py:242`),名字对不上就是找不到 `out/*.pth`。
- **`torch.compile` 会被静默关闭**:`apply_lora` 是往实例上 monkey-patch `forward`,和编译图不兼容,脚本在 `:155-157` 检测到 `--use_compile 1` 会打印提示并复位成 0;另外没有 GPU 时 `--device` 默认落到 cpu(`:76`)。README.md:766 提到 "train_lora.py 在 CPU 上通常也能比较轻快地完成",靠的就是这套小参数量 + 短序列配置。

## 下一步

LoRA 只是"注入领域知识";如果想让模型学会"两种回答里哪个更符合人的偏好",走 [05 DPO 偏好对齐](05_dpo.md)。
