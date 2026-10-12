# 08 Agentic RL / Tool Use(`trainer/train_agent.py`,485 行)

## 这一步在干什么(人话)

前面几步模型都是"一口气说完":给一句问题,生成一段回答,结束。这一步换成"边说边干活":模型先输出"我要用计算器算这个式子",程序**真的去执行**,把结果贴回对话,模型看到真实结果再想下一步,最后给答案。

关键就在"真的执行"。工具返回值不可能由模型编出来(某城市的实时温度、一段表达式的精确结果),只能由外部程序算好喂回上下文,模型才能基于它继续生成;所以训练时也必须把这套多轮循环真跑一遍,让模型习惯"我说一半 → 环境插一句事实 → 我接着说"这种结构。

答案对不对用规则判(比对数字或字符串),不靠人打分,这叫 RLVR(可验证奖励的强化学习)。产物:`out/agent_768.pth`。

## 先懂这几个词

| 术语 | 大白话 |
|---|---|
| 工具 / tool | 一段外部函数。模型只能"点菜"(说要调哪个、参数填什么),不能自己做菜 |
| 工具调用标签 | 模型写明"要调哪个工具、参数是什么"的固定包裹,程序靠正则把它抠出来 |
| 工具观测 | 工具执行结果,以 `role="tool"` 的消息回填进对话,模型下一轮读到的就是它 |
| ground truth / gt | 标准答案:既判最终答对没有,又被当作"该调几次工具"的期望数 |
| RLVR | 答案对错能用规则(字符串包含、数值比对)自动判定,不需要人或奖励模型打分 |
| rollout | 让当前模型完整走一遍对话,记录生成的 token 和当时的概率;这里是"生成→执行→回填→再生成"的多轮循环(上限 3 轮) |
| 组采样 `num_gen` | 同一道题连着做 4 份,看这一组里谁得分高就朝谁学(继承自 GRPO) |
| 优势 / 组内标准化 | 单条奖励减去同组均值再除标准差,变成"比同组平均好多少" |
| `thinking_ratio` | 按概率给上下文注入空思考标签,让模型学会"这题要不要先想一遍" |
| unfinished | 轮数用尽仍在调工具、没给出收尾答案的轨迹 |
| AST / 白名单 | 把表达式拆成语法树,只放行已知安全的节点,而不是直接 `eval` |

## 数据:一条样本长什么样

要解决的问题:RL 需要的不是"标准回答",而是"题目 + 可用工具清单 + 可验证的标准答案"——回答必须由模型自己生成。

- 两份 jsonl:`dataset/agent_rl.jsonl`(通用工具)/ `dataset/agent_rl_math.jsonl`(数学题,判分靠 `safe_math_eval`)。README 标注 86MB / 18MB,但本机 `dataset/` 下并没有这两个文件,需按 README 自行下载。
- 解析在 `AgentRLDataset`(`dataset/lm_dataset.py:230-256`):`parse_conversations()`(`:243-251`)从 system 消息的 `tools` 字段解析出工具清单,并丢掉最后一句 assistant(`messages[:-1]`)——标准回答不能当输入,否则没得可学;`__getitem__` 只返回 `messages / tools / gt`(`:256`),不含任何 token id。
- `collate_fn`(`train_agent.py:449`)原样保持嵌套结构打包:每条样本轮数不同,拼不成定长张量。

## 工具系统:模型说什么,程序怎么接

要解决的问题:得有一套"模型点得到、程序真跑得动"的工具,而且不能被模型写的怪东西搞死。

- 内置 6 个模拟工具(`MOCK_RESULTS`,`train_agent.py:56-63`):`calculate_math`、`unit_converter`、`get_current_weather`、`get_current_time`、`get_exchange_rate`、`translate_text`。都是查表/纯计算,不需要联网。
- `parse_tool_calls()`(`train_agent.py:76-81`):正则抓成对的工具调用标签,再 `json.loads` 里面的内容。
- `execute_tool()`(`train_agent.py:83-89`):按名字找函数,找不到返回 `None`,上层填 `{"error": "tool not found"}`(`:136`)。报错也是一种真实反馈,模型下一轮看得见。
- 参数填得对不对另有 `CHECK_ARGS`(`train_agent.py:66-73`)逐项校验,只用于算分,不影响执行。
- 数学表达式走 `safe_math_eval()`(`trainer/trainer_utils.py:183-209`):不用 `eval`(模型写什么就执行什么,太危险),而是解析 AST 后只放行白名单节点,其余一律 `ValueError`。幂运算统一过 `pow_guard`(`:185-187`),指数绝对值超 1e4、或结果量级超 10^100 直接拒绝,拦的就是 `9**9**9`、`10**99999` 这类能把进程算死的式子;另有表达式长度上限 512(`:205`)、全角符号与 `^ × ÷` 归一化(`:203`)。

## 多轮 rollout:生成 → 真执行 → 回填 → 再生成

这是全流程里唯一"环境"参与的地方:`rollout_single()`(`train_agent.py:92-157`)。

- 外层循环 `max_turns=3`(`:92` 函数默认,`:249` 硬编码传 3):每轮重新 `apply_chat_template` 拼上下文 → 采样(温度 0.8、`num_generations=1`,`:106-112`)→ 解析工具调用。没解析出调用就 `break`(`:125-126`),视为模型已答完。
- `open_thinking = random.random() < thinking_ratio`(`:100`):同题几份采样各自掷骰子,模型同时见到"带思考""不带思考"两种模板。
- 有调用:模型这段话先入 messages(`:128-129`),逐个执行工具,结果截断到 2048 字符(`:136`,防天文数字撑爆 tokenizer),以 `role="tool"` 追加(`:137`)。
- 回填的文本要重新 tokenize:`:139-152` 用临时 marker 差分出"环境插的这句"的 token id 追加到序列尾部,并把 mask 置 0(`:151`)——这些 token 不是模型说的话,不进损失,否则模型会把"某城市 22 摄氏度"当成自己的输出背下来。
- 轮数耗尽仍在调用 → `unfinished`(`:127`),带出去参与扣分。
- `rollout_batch()`(`train_agent.py:159-180`):每个 prompt 复制 `num_gen` 份独立跑(`:169`;`--num_generations` 默认 4,`:388`),同题 4 条轨迹互为对照组,这是组内标准化的前提。

## 奖励设计(`calculate_rewards`,`train_agent.py:188-239`)

要解决的问题:多轮轨迹没法用"和参考答案像不像"打分,只能拆成一堆规则小分项,最后加总并裁剪到 ±3(`:218`、`:238`)。按"有没有调工具"分成两条路。

| 项 | 分值 | 位置 |
|---|---|---|
| 工具调用标签开闭不配对 | −0.5 / 处 | `:201` |
| 无工具:答案长度 5~800 字符 | +0.5 / −0.5 | `:204` |
| 无工具:思考段长度 20~300 | +1.0 / −0.5 | `:207` |
| 无工具:思考闭合标签恰好 1 个 | +0.25 / −0.25 | `:208` |
| 无工具:奖励模型(RM)打分 | +score | `:210-216` |
| 两条路都有:重复惩罚 `rep_penalty` | 负分,单项上限 0.5 | `:217`、`:237`;函数 `:33` |
| 有工具:调用次数与 gt 的偏差 `tool_gap` | +0.5(为 0 时)/ −0.5×gap | `:230-231` |
| 有工具:最终答案命中 gt | +2.5 × 命中比例 | `:235`;判分函数 `validate_gt_in_text` `:183-186` |
| 有工具:轮数用尽未收尾 | −0.5 | `:236` |

两点值得注意:`gt` 在这条分支身兼两职,既是"该调几次工具"的期望次数(`:230`),又是"答案里该出现几个值"的分母(`:235`),所以数学数据的 gt 写法必须和期望调用次数对齐;奖励模型只在"没调工具"分支参与打分(`:210-216`),走工具路线后对错全由规则和 gt 决定,这就是 RLVR。

## 训练循环(`rl_train_epoch`,`train_agent.py:242-364`)

和 [07 GRPO / CISPO](07_grpo_cispo.md) 同一套骨架:组内标准化优势 → 对参考模型的 KL → clip 后的策略损失;唯一区别是采样换成带环境的多轮 rollout。

- `:248-249`:`torch.no_grad()` 下调 `rollout_batch`,单轮生成上限 `--max_gen_len`(默认 768,`:385`)
- `:252-261`:prompt 与 response 拼成一条序列,prompt 段 mask 补 0;超过 `--max_total_len`(默认 2500,`:386`)从左侧截断
- `:311-314`:按 `num_generations` reshape 成组,`(reward - 组均值)/(组标准差 + 1e-4)` 得优势。KL 系数 `--beta` 默认 0.1(`:389`),`--loss_type` 默认 `cispo`(`:390`),参考模型由同一份权重冻结而来(`:430-431`)
- 只有 mask 为 1 的 token 进损失(模型自己生成的部分),工具观测(`:151`)和 prompt 都排除
- `:361`:每 `--save_interval`(默认 10)步把新权重同步给 rollout 引擎(`:469` 开局先同步一次),否则引擎一直拿旧模型采样

## 运行

```bash
cd trainer
python train_agent.py --data_path ../dataset/agent_rl_math.jsonl
# rollout 换 sglang 加速(见 10_rollout_engine.md),多卡按 README 约定用 torchrun
python train_agent.py --rollout_engine sglang --sglang_base_url http://localhost:8998 --sglang_shared_path ./ckpt_mm --data_path ../dataset/agent_rl_math.jsonl
```

`--data_path` 默认是 `../dataset/agent_rl.jsonl`(`:387`)。`--rollout_engine` 默认 `torch`(`:402`),慢但零依赖;选 `sglang` 时 `:403`/`:405` 的地址与共享目录要和服务端对上。奖励模型在 `:433` 无条件加载,默认路径 `../../internlm2-1_8b-reward`(`:401`),本地没这份权重就起不来。

评测:`python scripts/eval_toolcall.py --weight agent`(`--weight` 默认 `full_sft`,不指定就评错模型)。

## 新手常见坑

- `thinking_ratio` 有两个值:函数默认 `0.5`(`:92`),但真正生效的是命令行默认 `0.1`(`:400`,经 `:249` 传入),即只有约 10% 的轨迹带思考模板。
- `max_turns` 不是命令行参数:函数默认 `3`(`:92`)、`:249` 又硬编码 `max_turns=3`,想要更长轨迹只能改这两处。
- 工具调用的 JSON 写坏会被**静默丢弃**(`:79-80` 是裸 `except: pass`):不报错,只是少算一次调用,表现为 `tool_gap` 莫名偏大、奖励偏低。先查模型输出的 JSON,别怀疑环境。
- 工具观测的 token 必须 mask 掉(`:151`);这段逻辑一旦被改动,模型会把环境返回的事实当自己的话学下来,从此"凭记忆报天气"。

## 下一步

轨迹跑通、奖励能自动验证之后,采样速度就成了瓶颈,看 [10 Rollout 引擎](10_rollout_engine.md);想确认模型是真会调工具而不是背答案,看 [11 评测与部署](11_eval_deploy.md)。