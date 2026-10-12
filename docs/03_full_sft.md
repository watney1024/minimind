# 03 全参监督微调 SFT(`trainer/train_full_sft.py`,164 行)

## 这一步在干什么(人话)

预训练出来的模型只会**续写**:你打"天空是",它接"蓝色的"——它从来没见过"一问一答"这种形式。

SFT(Supervised Fine-Tuning,监督微调)就是拿大量真实的问答对话再训一轮,把"续写机器"拧成"对话助手"。**全参**指模型的所有权重都参与更新(对比 [04 LoRA](04_lora.md) 只训一小段旁路)。

产出 `out/full_sft_768.pth`,也就是项目的 **Zero 对话模型**。这一步是全链路真正的分水岭:后面 LoRA、DPO、PPO、GRPO、Agentic RL、蒸馏**全部以它为起点**。

## 先懂这几个词

| 术语 | 大白话 |
|---|---|
| SFT / 监督微调 | 拿"标准问答对"继续训练,让模型学会按要求的形式回答 |
| 全参训练 | 模型的每一层权重都更新;与之相对的是只更新一小部分(LoRA) |
| 学习率 lr | 每次参数改动的幅度。太大不但学不好,还会把已经学会的东西冲坏 |
| 有效 batch | 真正参与一次参数更新的样本数 = `batch_size × accumulation_steps` |
| 灾难遗忘 | 用新数据猛训,会把旧能力覆盖掉。SFT 的学习率比预训练小得多,就是为了防这个 |

## 与预训练的差异(逐项对代码核实)

代码骨架确实与预训练同构:`train_epoch`(`:24-72`)里的前向、`res.loss + res.aux_loss`(`:35`)、累积、裁剪、step 逻辑与 `train_pretrain.py:24-71` 一致;主流程在 `if __name__ == "__main__":`(`:75`)到 `:164`。

但**"差异只在数据类"这个说法不准确**——默认超参也改了,而且改动很大:

| 参数 | Pretrain | Full SFT | 为什么要变 |
|---|---|---|---|
| 数据类 | `PretrainDataset` | `SFTDataset`(`dataset/lm_dataset.py:61`) | 核心差异,见下节 |
| 数据格式 | `{"text": ...}` | `{"conversations": [{"role","content"}, ...]}` | 需要多轮角色信息 |
| 数据文件 | `pretrain_t2t_mini.jsonl`(`train_pretrain.py:93`) | `sft_t2t_mini.jsonl`(`:94`) | |
| labels | 全序列(pad 外) | **只保留 assistant 回复段** | |
| `--learning_rate` | 5e-4 | **1e-5**(`:81`) | 小 50 倍。模型已经会说话了,这里是"塑形"不是"重教" |
| `--batch_size` | 32 | **16**(`:80`) | |
| `--accumulation_steps` | 8 | **1**(`:85`) | 于是有效 batch 从 256 降到 16 |
| `--max_seq_len` | 340(`train_pretrain.py:90`) | **768**(`:91`) | 一轮完整对话比一行语料长得多 |
| `--from_weight` | `none`(随机初始化) | **`pretrain`**(`:95`) | SFT 必须接着预训练权重训 |
| `--save_weight` | `pretrain` | `full_sft`(`:78`) | 决定产物名 `out/full_sft_768.pth` |
| `--epochs` | 2 | 2(`:79`) | 相同 |
| `--use_moe` / `--hidden_size` | 0 / 768 | 0 / 768(`:92`、`:89`) | 相同 |

> 上表的 lr、batch、累积步数三项是旧版文档漏掉的。只看代码骨架会误以为"改个数据类就行",照抄预训练的 5e-4 去跑 SFT 会把模型训坏。

## 真正决定能力的那一处:labels

差异表里其他项都是"配合数据格式",只有 labels 改变模型学到什么(实现见 [00 分词器与数据集](00_tokenizer_dataset.md) 的 `generate_labels`,`dataset/lm_dataset.py:91-107`):

- 整条 label 先全填 `-100`(`:92`)
- 扫描到 assistant 起始标记(`bos_id`,`:68`)后,才把这一段回答的 token 回填成 label(`:95-103`)
- user 提问、system 设定全程保持 `-100`,即 PyTorch 交叉熵会跳过它们

**预训练每个 token 都要预测下一个 token;SFT 只在"模型自己该说的话"上算 loss。** 同样是这些对话文本,前者学会续写,后者学会答话。

## Tool Call 与自适应思考在哪训

已并入主线 SFT 数据,`full_sft` 权重默认就带这两项基础能力:

- **工具调用**:数据里混有 OpenAI 风格的多轮消息,用 `tool_call` / `tool_response` 成对标记,和别的样本走同一套 chat template
- **自适应思考**:`post_processing_chat`(`dataset/lm_dataset.py:31-38`)以 80% 概率删掉模板生成的空思考段(`:33` 是那段字面量,`:35` 是概率),让模型同时见过"直接答"和"先想再答"两种示范;推理时可用 `--open_thinking 1` 强制思考
- 数据加载时显式声明 `features`(`dataset/lm_dataset.py:66`),容忍 jsonl 里 `tools` / `tool_calls` / `reasoning_content` 等可选字段缺失
- 专项评测见 [11 评估与部署](11_eval_deploy.md) 的 `scripts/eval_toolcall.py`

## 运行与验证

```bash
cd trainer                                     # 必须在此目录,见坑 1
python train_full_sft.py                       # 默认读 ../dataset/sft_t2t_mini.jsonl,基于 out/pretrain_768.pth
torchrun --nproc_per_node 4 train_full_sft.py  # 多卡 DDP
cd .. && python eval_llm.py --weight full_sft   # 冒烟测试:能按对话格式回答即算通
```

## 新手常见坑

1. **必须在 `trainer/` 目录下启动**。脚本里所有路径都相对该目录写死:权重目录 `--save_dir` 指 `../out`(`:77`)、数据 `--data_path` 指 `../dataset/...`(`:94`)、断点状态读写 `../checkpoints`(`:110`)。在仓库根目录直接跑会报找不到文件。
2. **不要把学习率改回 5e-4**。预训练那个量级对 SFT 太猛,会把已经学会的语言能力冲坏;默认 1e-5(`:81`)是有意的。
3. **显存占用与预训练不可直接比较**。虽然 batch 从 32 降到 16(`:80`),但 `max_seq_len` 从 340 涨到 768(`:91`),单步 token 总量反而略增(16×768 > 32×340)。
4. **必须先有 `out/pretrain_768.pth`**。`--from_weight` 默认是 `pretrain`(`:95`),`init_model` 会按此拼出 `out/pretrain_768.pth` 去加载;没跑过预训练就会在这里失败。

## 下一步

`full_sft` 是后面所有增强阶段的分叉点:想要更小代价做领域定制看 [04 LoRA](04_lora.md),想对齐人类偏好看 [05 DPO](05_dpo.md),想用奖励信号继续强化看 [06 PPO](06_rlaif_ppo.md) 与 [07 GRPO/CISPO](07_grpo_cispo.md)。
