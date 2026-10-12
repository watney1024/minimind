# MiniMind 训练链路总览(新手先读我)

> 行号引用已按当前工作区代码(`git HEAD = f659b55`)逐条打开文件核对。代码改动后行号会漂移,请以实际读到的内容为准。

面向零基础读者。看不懂术语就回到下面的词表查,每篇文档开头也各有一份自己的词表。

## 整条链路在干什么(人话)

把一个语言模型从零造出来,有点像培养一个人:先让他大量读书(预训练),再教他怎么跟人说话(指令微调 SFT),之后才谈得上补功课、练专项。

"大量读书"这一步只能让模型学会一件事:给定前半句话,猜下一个词最可能是什么。所以刚读完书的模型只会**续写**——你输入一句问话,它倾向于接着造出"长得像同类文本"的内容,而不是"回答你"。原因很简单:预训练语料里全是连续文本,模型从没见过"user 提问 / assistant 应答"这种对话格式,它不知道被问话时该怎么收束。

真正把"接龙机器"变成"能对话的助手"的,是 SFT:拿一批现成的问答对话喂给它,让它学会"看到 user 的话就该输出 assistant 的话"这个格式和习惯。

剩下的 LoRA / DPO / PPO / GRPO / 蒸馏等等,都不是必须步骤,而是**在这个会对话的基础上做不同方向的加强**:省显存地适配某个领域、让回答更符合人类偏好、让它学会先算账再调工具、或者把大模型的输出风格模仿到小模型上。

每个阶段的代码都在 `trainer/train_*.py` 里,长得几乎一模一样(共用一套骨架),所以学会读一篇就能读全部。

## 先懂这几个词

| 术语 | 大白话 |
|---|---|
| token / 分词器(tokenizer) | 把文字切成模型能处理的小块(字/词/片段),切法由一份"词表"决定 |
| 词表(vocab) | 这块小块的总目录,本仓库默认 6400 个(`model/model_minimind.py:18`) |
| 预训练(Pretrain) | 用海量纯文本让模型练"猜下一个词",只学语言和知识,不学对话 |
| SFT / 指令微调 | 用"问题→标准回答"的对话数据,教模型按人类期待的方式应答 |
| 全参微调 | 训练时模型所有参数都允许改动(对比之下 LoRA 只改一小撮附加参数) |
| LoRA | 冻结原模型,只在旁边加一小块可训练的"补丁",省显存、可插拔 |
| loss | 一个数字,表示"这次预测得有多差"。训练就是想办法让它变小 |
| 梯度 / 梯度下降 | 梯度指"往哪个方向调参数能让 loss 降得最快",梯度下降就是照着这个方向调 |
| 学习率(lr) | 每次调参的步子大小。太大学费,太小学得慢 |
| epoch / step / batch | epoch = 把整份数据过一遍;batch = 一次喂进去的样本数;step = 喂一批并更新一次参数 |
| 权重(`.pth`) | 训练好的模型本体(一大坨数字),保存成文件,下一阶段接着用 |
| hidden_size | 模型内部每个词用多长的向量表示,本仓库默认 768;它决定权重文件名里的数字 |
| MoE | 混合专家:一层里放多个"专家"网络,每次只激活其中几个。用 `--use_moe 1` 打开 |
| DPO / PPO / GRPO / CISPO | 都属"对齐/强化"类方法:不再只模仿标准答案,而是根据"这份回答好不好"的信号(偏好对或奖励分数)调模型 |
| 奖励模型(reward model) | 一个专门给回答打分的模型,充当 RL 阶段的"阅卷老师" |
| 蒸馏 | 让学生模型去模仿教师模型的输出(或教师的概率分布),从而小模型学到本事 |
| 工具调用 / Agentic | 让模型学会输出结构化的"我要调用哪个函数、参数是什么",由外部执行后把结果喂回去 |
| DDP | 数据并行:多张卡各拿一份模型副本、各算一部分数据,再同步梯度 |
| 混合精度(bf16/fp16) | 用更短的数存中间计算,省显存、跑得快,代价是精度略低 |
| checkpoint / 断点续训 | 把"训练到第几步、优化器状态"也存下来,崩了能接着练而不是从头再来 |
| rollout | 让模型真的把一段回答生成出来(RL 阶段要先有生成结果,才能打分并据此更新) |

## 先读我:建议阅读顺序

`docs/` 下 00~11 共 12 篇编号文档 + 本篇总览。零基础的读法:

| 顺序 | 文档 | 它解决什么问题 |
|---|---|---|
| 1 | [00 分词器与数据集](00_tokenizer_dataset.md) | 文字怎么变成数字喂进模型;每个阶段分别要哪种数据文件 |
| 2 | [01 模型结构](01_model_architecture.md) | attention、MoE、层数这些"骨架"代码在哪、怎么搭起来 |
| 3 | [02 预训练](02_pretrain.md) | 第一个真正跑起来的训练:怎么只靠续写任务把语言规律压进参数 |
| 4 | [03 全参 SFT](03_full_sft.md) | 从"只会续写"变成"会对话"的那一步,也是后面所有阶段的起点权重 |
| 5 | [04 LoRA](04_lora.md) | 显存不够 / 想按领域定制时,只训一小块补丁怎么做 |
| 6 | [05 DPO](05_dpo.md) | 不训奖励模型,直接用"更好/更差"两个回答的对比来校正偏好 |
| 7 | [06 RLAIF 与 PPO](06_rlaif_ppo.md) | 引入奖励模型 + critic 网络,用 PPO 的目标函数做强化学习 |
| 8 | [07 GRPO 与 CISPO](07_grpo_cispo.md) | 不要 critic、改用一组采样互相比较来算优势;以及 CISPO 的裁剪方式 |
| 9 | [08 Agentic RL](08_agentic_rl.md) | 让模型学会多轮工具调用,并用规则算出的结果当奖励 |
| 10 | [09 蒸馏](09_distillation.md) | 教师模型的输出/分布怎么变成学生模型的 loss |
| 11 | [10 Rollout 引擎](10_rollout_engine.md) | "让模型生成长回答"这件事为什么要单独抽出来,以及 torch / sglang 两种实现 |
| 12 | [11 评估与部署](11_eval_deploy.md) | 训好的权重怎么对话、怎么转成 transformers 格式、怎么开 API |

只看一篇就上手的话:**读 02 和 03**。这两篇覆盖了必需路径,其余都是可选增强。

## 整体路线图

```
随机初始化(什么都没有)
   ↓  预训练 train_pretrain.py --from_weight none
out/pretrain_768.pth   ← 只会"续写",不会对话
   ↓  全参 SFT train_full_sft.py --from_weight pretrain
out/full_sft_768.pth   ← 会对话了。这是后面所有增强阶段的公共起点
   ↓  任选其一/多个(都从 full_sft 出发,互不依赖)
   ├─ LoRA        → out/lora_medical_768.pth   (领域补丁,推理时叠加)
   ├─ DPO         → out/dpo_768.pth            (偏好对齐)
   ├─ PPO         → out/ppo_actor_768.pth      (奖励模型打分的 RL)
   ├─ GRPO/CISPO  → out/grpo_768.pth           (组内比较的 RL)
   ├─ Agentic RL  → out/agent_768.pth          (工具调用 RL)
   └─ 蒸馏        → out/full_dist_768.pth      (教师→学生)
   ↓
评估 / 部署 eval_llm.py、scripts/
```

三条要点:

1. **权重是接力棒**。每个脚本靠 `--from_weight` 指定"我站在谁的肩膀上",命名规则见下面产物表。所以顺序不能乱:没跑过预训练,SFT 就找不到输入权重。
2. **`none` 是唯一真正从零开始**。`--from_weight none` 时不加载任何权重,模型参数是随机数(`trainer/trainer_utils.py:125`),这时它输出的东西基本是乱码。
3. **增强阶段之间通常不叠加**。DPO、PPO、GRPO、Agent、蒸馏都是从 `full_sft` 各走各的路,产出各自的权重;要合并效果得自己设计流程。

## Pipeline 与产物

| 阶段 | 脚本 | 输入权重(`--from_weight`) | 输出(`out/`) | 文档 |
|---|---|---|---|---|
| 分词器 | `trainer/train_tokenizer.py` | - | 写到 `model_learn_tokenizer/`,不进 `out/`。脚本首行明确不建议重训:仓库已自带 `model/tokenizer.json` + `model/tokenizer_config.json` | [00](00_tokenizer_dataset.md) |
| 模型结构 | `model/model_minimind.py` | - | - | [01](01_model_architecture.md) |
| 预训练 | `trainer/train_pretrain.py` | `none`(随机初始化,`:94`) | `pretrain_768.pth` | [02](02_pretrain.md) |
| 全参 SFT | `trainer/train_full_sft.py` | `pretrain`(`:95`) | `full_sft_768.pth` | [03](03_full_sft.md) |
| LoRA | `trainer/train_lora.py` | `full_sft`(`:88`) | `lora_medical_768.pth`(文件名用 `--lora_name`,`:61`) | [04](04_lora.md) |
| DPO | `trainer/train_dpo.py` | `full_sft`(`:141`) | `dpo_768.pth` | [05](05_dpo.md) |
| PPO(RLAIF) | `trainer/train_ppo.py` | `full_sft`(`:336`) | `ppo_actor_768.pth`(`:309`;critic 只进 resume 文件,不进 `out/`) | [06](06_rlaif_ppo.md) |
| GRPO / CISPO | `trainer/train_grpo.py` | `full_sft`(`:224`) | `grpo_768.pth`(`:202`;注意 `--loss_type` 的默认值其实是 `cispo`,`:221`) | [07](07_grpo_cispo.md) |
| Agentic RL | `trainer/train_agent.py` | `full_sft`(`:393`) | `agent_768.pth`(`:370`) | [08](08_agentic_rl.md) |
| 蒸馏 | `trainer/train_distillation.py` | `--from_student_weight` / `--from_teacher_weight`,均默认 `full_sft` | `full_dist_768.pth`(按学生模型的 hidden_size / moe 命名,`:124-125`) | [09](09_distillation.md) |
| Rollout 引擎 | `trainer/rollout_engine.py` | - | - | [10](10_rollout_engine.md) |
| 评估 / 部署 | `eval_llm.py`(仓库根)、`scripts/` | 任意权重前缀 `--weight` | - | [11](11_eval_deploy.md) |

权重文件名的统一规则:`{权重前缀}_{hidden_size}{moe后缀}.pth`,其中 `moe后缀 = '_moe'`(开了 `--use_moe 1`)或空串。依据是同一份逻辑出现在 `trainer/trainer_utils.py:67-69`(存)、`:126-127`(读)以及各脚本的保存块(如 `train_pretrain.py:61-62`)。本仓库 `hidden_size` 默认 768,所以 dense 模型就是 `pretrain_768.pth`、MoE 模型就是 `pretrain_768_moe.pth`。`out/` 下现有的 `pretrain_768.pth` / `full_sft_768.pth` 即由此规则产出。

`scripts/` 下的工具:`web_demo.py`(网页对话)、`serve_openai_api.py`(起一个 `/v1/chat/completions` 兼容服务,`:178`)、`chat_api.py`(调 OpenAI 兼容接口的客户端)、`eval_toolcall.py`(工具调用评测)、`convert_model.py`(torch 权重 ↔ transformers 格式,并可合并 LoRA)。

各阶段默认数据文件(`--data_path`):预训练 `dataset/pretrain_t2t_mini.jsonl`、SFT 与蒸馏 `dataset/sft_t2t_mini.jsonl`、DPO `dataset/dpo.jsonl`、PPO 与 GRPO `dataset/rlaif.jsonl`、LoRA `dataset/lora_medical.jsonl`、Agentic RL `dataset/agent_rl.jsonl`。

## 公共训练基建(`trainer/trainer_utils.py`)

所有训练脚本共享同一套骨架。这里每个函数解决的是"每个阶段都要写一遍、写错一次就毁掉一整轮训练"的琐事。main 函数的 9 步骨架以 `trainer/train_pretrain.py` 为参照:`:75-99` 是命令行参数,`:101-164` 依次是下面 9 步。

- **怎么开始多卡**:`init_distributed_mode()`(`trainer_utils.py:46`)读 `RANK`/`LOCAL_RANK` 环境变量:没有 `RANK` 就直接返回 0 当普通单进程训练(`:47-48`),用 `torchrun` 起才会初始化 NCCL 进程组并绑卡(`:50-52`)。所以同一份脚本单卡多卡都能跑。
- **学习率怎么随训练衰减**:`get_lr()`(`trainer_utils.py:42`)是**余弦退火**,不是线性:`lr * (0.1 + 0.45 * (1 + cos(pi * step/total)))`。起步等于 `lr`,收尾停在 `0.1 * lr`,中间先慢后快地降。
- **权重从哪来**:`init_model()`(`trainer_utils.py:121`)先无条件按配置建一个新模型(`:123`),只有 `from_weight != 'none'` 时才去 `out/{from_weight}_{hidden_size}{moe后缀}.pth` 读权重(`:125-129`);分词器固定从 `--tokenizer_path`(默认 `../model`)加载(`:122`)。注意加载用 `strict=False`(`:129`),形状对不上也不报错——见下面的坑。
- **权重和进度怎么存**:`lm_checkpoint()`(`trainer_utils.py:65`)一次写两个文件:`checkpoints/{weight}_{hidden_size}{moe后缀}.pth`(纯权重,`:68`)和 `..._resume.pth`(权重 + optimizer + epoch + step + world_size + wandb id,`:69`、`:87-106`)。落盘前所有张量转 fp16 并存到 CPU(`:75`),所以 768 dim 的 dense 权重约 137MB。写文件走"先 `.tmp` 再 `os.replace`"(`:76-78`、`:104-106`),中途断电不会留下半个坏文件。
- **换卡数继续练**:`lm_checkpoint` 加载分支发现保存时的 GPU 数和现在不一样,会按 `step * 旧卡数 // 新卡数` 换算步数(`:112-116`),这样 `--from_resume 1` 在 4 卡跑一半换 2 卡也不会跳步或重复。
- **恢复后别重做已训练的 batch**:`SkipBatchSampler`(`trainer_utils.py:136`)在迭代器里丢掉前 `skip_batches` 个 batch(`:142-155`),配合 `train_pretrain.py:152-153` 使用;它同时重写了 `__len__`(`:157-159`)。
- **混合精度**:默认 `--dtype bfloat16`(`train_pretrain.py:82`);`:112-114` 按设备决定 `autocast` 的 dtype,CPU 设备直接退化成 `nullcontext`(即不开混合精度)。fp16 时另配 `GradScaler`(`:129`,`enabled=(dtype=='float16')`)。
- **MoE 的额外 loss**:总 loss 一律是 `res.loss + res.aux_loss`(如 `train_pretrain.py:35`)。`aux_loss` 是 MoE 路由的负载均衡项,防止所有 token 都挤到同一个专家;dense 模型下它为 0,加上去不影响结果。日志里会把两者分开打印(`train_pretrain.py:51-56`)。
- **谁来给回答打分**:`LMForRewardModel`(`trainer_utils.py:162`)加载 InternLM2 的 reward 模型(`trust_remote_code=True`,`:164-166`),`get_score()`(`:170`)把"对话历史 + 新问题 + 待评回答"拼成它的评分格式,返回裁剪到 `[-3, 3]` 的分数(`:178-179`)。PPO / GRPO / Agentic RL 三个脚本都用它(默认权重路径 `../../internlm2-1_8b-reward`,见 `train_ppo.py:337`、`train_grpo.py:225`、`train_agent.py:401`)。
- **模型写的算式能不能真算**:`safe_math_eval()`(`trainer_utils.py:183`)用 AST 白名单求值代替 `eval`,只放行四则运算、`pow` 和一批 `math` 函数/常量(`:199-202`),`pow_guard`(`:185-187`)专门拦 `9**9**9`、`10**99999` 这类把进程算死的输入,表达式长度上限 512(`:205`),还会把 `^`、`×`、`²`、全角括号等模型爱写的符号先归一化(`:203`)。Agentic RL 的 `calculate_math` 工具就是它(`train_agent.py:57`)。

## 前置条件

这些条目都能在代码或根 `README.md` 里查到出处;查不到的(比如"总共要跑多少小时")我就不给数字。

- **硬件**:作者用 8×RTX 3090(24GB)/ CUDA 12.2 / Python 3.10.16 做参考配置(`README.md:217-222`)。所谓"2 小时、3 块钱"指的是 **SFT 阶段单张 3090 跑完 1 epoch** 的实测耗时与对应租卡成本(`README.md:42`),不是整条链路的时间。默认参数下预训练是 2 epoch(`train_pretrain.py:78`)、SFT 是 2 epoch(`train_full_sft.py:79`),请照这个量级自己估。
- **能不能纯 CPU 跑**:参数层面允许——`--device` 在没有 CUDA 时自动落到 `cpu`(`train_pretrain.py:81`),CPU 下 autocast 关掉(`:114`);根 `README.md:283-284` 也说明可以按设备选 CPU 或 MPS,但"训练速度与兼容性会有非常大的差异"。也就是说 CPU 适合读懂代码和短样本试跑,不适合真跑完一个 epoch。
- **数据要下载**:仓库里 `dataset/` 只有代码,没有语料。默认只需 `pretrain_t2t_mini.jsonl` 与 `sft_t2t_mini.jsonl` 两个文件就能跑通"预训练 + SFT"对话模型(`README.md:291-294`);本机实测量级分别约 1.2GB 与 1.7GB。下载命令:`modelscope download --dataset gongjy/minimind_dataset <文件名> --local_dir ./dataset`,下好的文件必须放在 `./dataset` 目录(`dataset/dataset.md`)。
- **torch 不在依赖表里**:`requirements.txt` 钉死了 `transformers==4.57.6`、`trl==0.13.0`、`numpy==1.26.4` 等版本,但 `torch` / `torchvision` 两行是注释掉的(`requirements.txt:33-34`)。必须自己按显卡驱动装对应 CUDA build——这是下面第一个坑的根源。
- **RL 阶段额外依赖**:PPO / GRPO / Agentic RL 需要一个外部奖励模型(InternLM2-1.8B reward),不在本仓库里,默认从 `../../internlm2-1_8b-reward` 加载,要事先下好或用 `--reward_model_path` 指对路径。
- **DPO / LoRA / Agentic 的数据不在默认下载清单里**:`dpo.jsonl`、`lora_medical.jsonl`、`agent_rl.jsonl` 是各脚本的默认 `--data_path`,但当前 `dataset/` 目录里确实没有这几个文件,要做这些阶段得另外下载(本机实测:`pretrain_t2t_mini`、`sft_t2t_mini`、`rlaif` 在,其余缺)。
- **执行目录**:所有训练脚本必须在 `trainer/` 目录下执行(`README.md:671`),因为 `--save_dir ../out`、`--data_path ../dataset/...`、`save_dir='../checkpoints'` 全是相对路径。反过来,`eval_llm.py` 在仓库根执行(它默认 `--load_from model`、`--save_dir out`,`eval_llm.py:34-35`)。

## 环境与运行(踩坑记录)

- conda 环境 `minimind`(Python 3.11)。仓库根 `AGENTS.md` 记录本机 NVIDIA 驱动为 535.x(最高支持 CUDA 12.2),**torch 必须装 cu121 或更旧的 build**(如 `torch==2.4.1+cu121`);pip 默认装的新版 cu130 build 会**静默回退 CPU**:GPU 占用 0%、CPU 满载、不报任何错。装完先跑 `python -c "import torch; print(torch.cuda.is_available())"` 确认。
- `datasets==3.6.0` 要配 `pyarrow==21.0.0`:新版 pyarrow 要求 numpy>=2.0,与钉死的 numpy 1.26.4 冲突,一 import 就 ImportError。
- `ppqenv`(Python 3.9 + PPQ 0.6.6)是量化专用环境,缺 transformers/datasets,不要拿它跑本项目。
- 多卡:在 `trainer/` 目录下 `torchrun --nproc_per_node N train_xxx.py`(`README.md:353`;`N>=2` 才真正进 DDP,`:680` 给的示例是 `--nproc_per_node 1`,效果等同单卡)。断点续训:`--from_resume 1`。

## 新手常见坑

1. **看 loss 判断是否在训练前先确认 GPU**。混合精度/新 torch build 的问题不报错,只在日志的 `epoch_time` 上暴露。判据:`torch.cuda.is_available()` 为 `False` 时 `--device` 会静默切到 CPU(`train_pretrain.py:81`)。
2. **在仓库根直接 `python trainer/train_pretrain.py` 不会立刻报错说"目录不对"**。默认路径全是相对当前工作目录的(`--save_dir ../out`、`--data_path ../dataset/...`,`train_pretrain.py:76`、`:93`;分词器路径 `../model` 则是 `init_model` 的默认值,`trainer_utils.py:121-122`)。所以它会先在仓库**外面**静默建出一个 `../out` 目录(`train_pretrain.py:107` 的 `os.makedirs`;加了 `--from_resume 1` 还会再建 `../checkpoints`,`trainer_utils.py:66`),然后在加载分词器那一步才失败(`../model` 不存在,`:122`);就算你把分词器路径指对了,下一步读 `../dataset/pretrain_t2t_mini.jsonl` 同样会失败(`train_pretrain.py:127` → `dataset/lm_dataset.py:45`)。正确做法:`cd trainer` 再跑(根 `README.md:671` 就是这么要求的)。
3. **`--from_weight` 写错名字不会给你友好提示**。权重路径由 `trainer_utils.py:127` 拼字符串得到,拼错就是文件不存在;而加载用的 `strict=False`(`:129`)意味着文件存在但形状/键名不匹配时**会静默跳过对应参数**,你以为在续训、其实一半是随机初始化。改 `--hidden_size` / `--use_moe` 做实验时最容易踩到,留意 `init_model` 打印的参数量是否和你预期一致(`:131-132`)。
4. **`out/` 与 `checkpoints/` 别搞混**。`out/` 是各阶段产物(给下一阶段当输入、给 `eval_llm.py` 加载);`checkpoints/` 是断点续训状态(带 `_resume` 的还包含优化器,体积大得多,本机实测 649MB vs 137MB)。只删 `checkpoints/` 不影响已有权重,但会丢掉续训进度。

## 下一步

从 [00 分词器与数据集](00_tokenizer_dataset.md) 开始顺序往下读;只想先跑起来看效果,就直接跳到 [02 预训练](02_pretrain.md) 和 [03 全参 SFT](03_full_sft.md)。
