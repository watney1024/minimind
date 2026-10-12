# 00 分词器与数据集

## 这一步在干什么(人话)

模型只认数字,不认文字。**分词器(tokenizer)** 就是那本"文字 ↔ 数字"的对照词典:把一句话切成若干片段(token),每个片段换一个编号。

但光有词典不够。训练前还得决定**每次喂给模型的东西长什么样、"标准答案"标在哪几个位置**——这是数据集类的活。这一步看着最没技术含量,却决定了后面一切:`labels` 标在哪,模型就只学哪里。预训练和 SFT 的本质差别不在模型、不在算法,就在这一篇里两个数据集类怎么造 labels。

## 先懂这几个词

| 术语 | 大白话 |
|---|---|
| token | 文本的最小片段,可能是一个汉字、一个英文词,也可能是词的一部分或标点 |
| 词表(vocab) | 所有 token 与编号的对照表,本项目 6400 个 |
| BPE | 造词表的算法:从单字出发,反复把"最常一起出现"的片段合并成一个新 token |
| 特殊 token | 人为塞进词表的记号,如标记"这句话是用户说的""回答到此结束" |
| padding | 一个 batch 里句子长短不一,短的补无意义 token 凑成同一尺寸才能打包计算 |
| label | "下一个 token 该预测成什么"的标准答案,与输入同长、向右错一格 |
| `-100` | label 里的占位值。PyTorch 交叉熵默认忽略编号 -100 的位置,写 -100 就是"这里别学" |
| chat template | 把 `[{role, content}, ...]` 消息列表拼成一段带特殊标记的纯文本的规则 |
| bos / eos | 一段内容的起止标记。本项目 bos 是 `<\|im_start\|>`(编号 1)、eos 是 `<\|im_end\|>`(编号 2),已从 `model/tokenizer.json` 核实 |

## 1. 分词器本身(`trainer/train_tokenizer.py`,189 行)

> 项目已自带训练好的 tokenizer(`model/tokenizer.json`)。文件头注释 `train_tokenizer.py:1-2` 明确不建议重训:换一本词典,输出就跟社区全对不上。此脚本仅供学习参考。

- `get_texts()`(`:13`):解决"从哪弄语料喂给 BPE"。逐行读 jsonl,兼容两种格式——pretrain 的 `{"text": ...}` 与 SFT 的 `{"conversations": [...]}`(`:30-34`)
- `train_tokenizer()`(`:45`):解决"词表怎么自动造出来"。用 HuggingFace `tokenizers` 库训练 **BPE**
  - `Tokenizer(models.BPE())`(`:46`)+ `pre_tokenizers.ByteLevel`(`:47`):先做字节级切分。任何字符都能拆成字节,所以不存在"这个词词表里没有"的问题
  - 特殊 token 共 36 个(`SPECIAL_TOKENS_NUM`,`:10`),构建于 `:49-63`:对话起止标记、`<tool_call>` / `<tool_response>` / `<think>` 等功能标记,再用 `<|bufferN|>` 补齐到 36 个(`:61-63`)——预留槽位,以后加标记不必重训
  - `trainers.BpeTrainer`(`:64-69`):`vocab_size` 取 `VOCAB_SIZE = 6400`(`:9`,与模型输出层宽度对齐;模型侧见 `model_minimind.py:16`)
  - `:75-77` 写出 `tokenizer.json`;`:99-126` 写出 `tokenizer_config.json`,其中 `chat_template` 在 `:121`——**对话拼接规则的真正定义处**,后面所有阶段都靠它
  - 入口 `__main__`(`:187-189`):训练 + 自检各一步
- `eval_tokenizer()`(`:129`):解决"词典造出来好不好用"。检查编解码一致性、中英文压缩率(字符数 ÷ token 数,越大越省)、流式解码时的字节缓冲

## 2. 数据集类(`dataset/lm_dataset.py`,260 行)

全是 PyTorch 原生 `Dataset`(**不做离线预处理,分词发生在运行时 `__getitem__`**,基类 `torch.utils.data.Dataset` 从 `:1` 导入)。代价是每步都要现场分词、吃 CPU;好处是改逻辑不用重新预处理 14GB 数据。

### PretrainDataset(`:40`)——只学"续写"

- `load_dataset('json', ...)`(`:45`),只用 `text` 字段
- `__getitem__`(`:50-58`):分词并截断到 `max_length-2`(`:52`),拼成 `[bos] + tokens + [eos]`(`:53`),右侧 pad 到定长(`:54`)
- **labels 怎么来**:`labels = input_ids.clone()`(`:56`),只把 pad 位置置 `-100`(`:57`)

也就是**每个真实 token 都要预测下一个 token**——预训练的全部目标。

### SFTDataset(`:61`)——只学"说",不学"听"

- `__init__`(`:62-69`):显式声明 `features`(`:66`,conversations 含 `role/content/reasoning_content/tools/tool_calls`),避免不同 jsonl 字段不一致直接报错
- `pre_processing_chat()`(`:9-29`):以约 20% 概率随机插一条 system prompt(`:26-28`,候选池 `:13-24`),让模型别只会"无设定对话";带 `tools` 的工具调用样本原样保留不处理(`:11`)
- `create_chat_prompt()`(`:74-89`):把 messages 交给 `tokenizer.apply_chat_template`(`:84-89`,带 `tools` 参数)拼成纯文本
- `post_processing_chat()`(`:31-38`):模板总会生成一段空的 think 标签(`:33` 是那段字面量),这里以 **80% 概率删掉它**(`:35`,判据是 `random() > 0.2`)。
  为什么这么干:模型同时看到"提问 → 直接回答"和"提问 → 思考 → 回答"两种示范,才学会**简单问题不思考**;否则每题都先空想一遍。这就是"自适应思考"能力的数据来源
- `generate_labels()`(`:91-107`):**最关键的一段**。整条 label 先全填 `-100`(`:92`),然后扫描 assistant 起始标记 `bos_id`(即 `<|im_start|>assistant`,构造于 `:68`),从该标记之后一直找到回答结束标记(`eos_id`,`:69`),**只把这一段回复的 token 回填成 label**(`:95-103`)

user 提问、system 设定全程保持 `-100`。于是同样的对话数据,预训练是"每个字都学",SFT 是"只学模型自己该说的话"。

### DPODataset(`:125`)——成对的"更好/更差"回答

- `__getitem__`(`:138-178`):把 `chosen`(更好的回答)和 `rejected`(更差的)各自套模板(`:142-148`),**同一随机扰动作用于两边**(`:149-151`)——两边删不删空 think 必须一致,否则差异就不只剩回答本身
- 返回 `x_chosen/y_chosen/mask_chosen/x_rejected/y_rejected/mask_rejected` 六个字段(`:171-178`)
- `generate_loss_mask()`(`:180-196`):和 SFT 的 `generate_labels` 同一套定位逻辑,但产出 0/1 掩码而不是 token 标签

> 订正:`__getitem__` 只**分别**返回两半。"前一半 chosen、后一半 rejected"的大 batch 是在训练脚本里 `torch.cat` 拼出来的(`trainer/train_dpo.py:63-65`)。

### RLAIFDataset(`:199`)——只给问题,答案模型自己生成

强化学习阶段没有标准答案,所以只要 prompt:

- `create_chat_prompt()`(`:212-220`)只取 `conversations[:-1]`(丢掉最后一轮标准回答)且 `add_generation_prompt=True`(`:215-220`,让文本停在"assistant 该说话了")
- `thinking_ratio=0.5`(`:200` 声明、`:204` 存下、`:214` 使用):按概率决定是否走 `open_thinking=True`

### AgentRLDataset(`:230`)——带工具调用轨迹的多轮对话

- `parse_conversations()`(`:243-252`):从 system 消息里解出工具定义 `tools`(`:248-249`),返回 `messages[:-1]`(`:251`)——**去掉 assistant 的最终轨迹,只留到用户提问为止**,工具调用序列由模型现场生成
- `__getitem__`(`:253-256`):返回 `messages / tools / gt`,`gt` 是 ground truth(标准答案与应有工具调用),用于规则判分

## 3. 数据文件(本机 `dataset/` 实测)

| 文件 | 实际大小 | 用于 |
|---|---|---|
| `pretrain_t2t_mini.jsonl` | 1.2 GB | 快速预训练 |
| `pretrain_t2t.jsonl` | 8.3 GB | 主线预训练 |
| `sft_t2t_mini.jsonl` | 1.7 GB | Zero 模型 SFT(已混 tool call) |
| `sft_t2t.jsonl` | 14.1 GB | 完整 SFT |
| `rlaif.jsonl` | 24 MB | PPO/GRPO/CISPO |
| `dpo.jsonl` | **本机缺**,需下载 | DPO 偏好对 |
| `agent_rl.jsonl` / `agent_rl_math.jsonl` | **本机缺**,需下载 | Agentic RL |

## 新手常见坑

1. **别重训 tokenizer**。`train_tokenizer.py:1-2` 明确写了原因:社区模型都基于现有词典,换词典后权重无法复用。
2. **把 `DATA_PATH` 指向预训练语料要留意**。该脚本历史上只解析 `conversations` 格式,指向 `{"text": ...}` 的预训练文件会一条文本都取不到,BPE 仅凭字节表训练出一个空词表,**而且全程不报错**(`:17-19` 记录了这次改动,现在取不到文本会直接抛错,`:38-43`)。
3. **pad 与 unk 是同一个 token**。`tokenizer_config.json` 里 `pad_token` 与 `unk_token` 都是 `<|endoftext|>`(`train_tokenizer.py:110` 与 `:113`)。上面那句 "pad 位置标 `-100`,所以 pad 不会被学" 成立的前提是标签屏蔽,而不是 pad 本身无害。
4. **运行时分词吃 CPU**。数据集类没有离线预处理,`__getitem__` 现场分词(`:52`、`:114`);CPU 满载而 GPU 空闲通常是这里,不是模型问题。

## 下一步

有了词典和数据格式,接下来要看"要训练的东西长什么样" → [01 模型结构](01_model_architecture.md)。
