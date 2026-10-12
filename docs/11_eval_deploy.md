# 11 评估与部署

## 这一步在干什么(人话)

模型训完只是磁盘上一堆数字,得先知道它会不会说话、说得好不好,再想办法让别人用得上它。
本篇按"由近到远"排:先在终端里跟它聊两句(`eval_llm.py`),再专测它会不会调用工具(`eval_toolcall.py`),然后把训练用的原生权重转成通用工具认得的格式(`convert_model.py`),最后对外提供服务 —— 本项目自带的 OpenAI 兼容 HTTP 接口、网页聊天界面,或交给 vllm / sglang / ollama 这类第三方推理引擎跑。

## 先懂这几个词

| 术语 | 大白话 |
|---|---|
| 推理 / 生成 | 让模型把回答一个字一个字吐出来,不改参数 |
| CLI | 命令行工具:敲一条命令、在终端里看结果,没有图形界面 |
| 多轮对话 / 历史轮数 | 把之前的问答一起再喂回去,模型才"记得"上文 |
| chat template | 对话模板:把"谁说了什么"拼成模型真正吃的字符串(角色标签、特殊符号靠它加),存在分词器目录的 `tokenizer_config.json` 里 |
| 流式输出 / stream | 不等整段写完,生成一点推一点,看到的是一句句冒出来 |
| HTTP API | 程序之间靠网页请求互相调用:你 POST 一段 JSON,我回一段 JSON |
| OpenAI 兼容接口 | 照抄 OpenAI 那套请求/返回格式(路径 `/v1/chat/completions`、字段 `messages`/`choices`),现成聊天前端不改代码就能接你自己的模型 |
| 端口 | 一台机器上区分不同网络服务的编号,8998 上的服务和 8000 上的服务互不干扰 |
| `.pth` | PyTorch 原生权重文件(一堆张量),必须配合本项目的模型代码才认得 |
| HF 格式 / transformers 目录 | HuggingFace 通用格式:一个文件夹放 `config.json` + 权重文件 + 分词器 + 对话模板,`transformers`/vllm/sglang/ollama 等都能直接加载 |
| safetensors | 一种权重文件格式,加载时不会执行任意代码,比老的 pickle 方式安全 |
| Tool Call | 模型不直接答,而是输出"我要调哪个函数、参数是什么"的 JSON,由外部真的执行后把结果回喂给它 |
| Streamlit | 用纯 Python 脚本快速搭网页界面的库,`streamlit run xxx.py` 就起一个网页 |

## CLI 快速测试:`eval_llm.py`(根目录)

问题:训完一版权重,最快怎么确认它没崩、还能不能说人话?

### 参数(`eval_llm.py:33-49`)

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--load_from` | `model` | 模型来源:名字里含 `model` 走原生 `.pth`,其它路径按 transformers 目录加载 |
| `--save_dir` | `out` | 原生权重所在目录 |
| `--weight` | `full_sft` | 权重文件前缀名,可选值见下 |
| `--lora_weight` | `None` | 额外挂一个 LoRA 权重(见 [04 篇](04_lora.md)) |
| `--hidden_size` / `--num_hidden_layers` | `768` / `8` | 模型规模,必须和训练时一致,否则加载报错 |
| `--use_moe` | `0` | 为 1 时权重文件名带 `_moe` 后缀 |
| `--inference_rope_scaling` | `False` | 位置编码外推开关,想超出训练长度推理时加(README 的「Ⅳ RoPE长度外推」一节) |
| `--max_new_tokens` | `8192` | 单次生成上限(不等于模型真实长文能力) |
| `--temperature` / `--top_p` | `0.85` / `0.95` | 采样随机性,越大越放飞 |
| `--open_thinking` | `0` | 为 1 时开启显式思考(模板预留 `<think>` 结构,README 的「5.2 Adaptive Thinking」一节) |
| `--historys` | `0` | 携带几轮历史对话(需偶数,0 = 每轮都当新对话) |
| `--show_speed` | `1` | 打印生成速度 tokens/s |
| `--device` | 有 GPU 用 `cuda`,否则 `cpu` | 运行设备 |

`--weight` 填各训练脚本存下来的前缀名(默认值取自各自的 `--save_weight`):`pretrain`(`trainer/train_pretrain.py:77`)、`full_sft`(`trainer/train_full_sft.py:78`)、`dpo`(`trainer/train_dpo.py:125`)、`ppo_actor`(`trainer/train_ppo.py:309`)、`grpo`(`trainer/train_grpo.py:202`)、`agent`(`trainer/train_agent.py:370`)、`full_dist`(`trainer/train_distillation.py:141`);LoRA 走 `--lora_weight`(如 `lora_medical`/`lora_identity`,`trainer/train_lora.py:72`)。脚本 help 里列的名字(`eval_llm.py:36`)与训练脚本实际存的不完全一致,以 `out/` 下真实文件名为准。

### 加载逻辑(`eval_llm.py:12-30`)

问题:同一个入口要能读原生 `.pth`,也要能读 HF 目录。

- 分支依据是字符串包含判断 `if 'model' in args.load_from`(`:14`)。命中走原生路线:自己 `MiniMindForCausalLM(MiniMindConfig(...))` 建结构(`:15-20`),再按 `./{save_dir}/{weight}_{hidden_size}{_moe}.pth` 拼路径加载(`:21-23`,`strict=True`,参数名必须完全对得上)。
- 挂 LoRA 时先 `apply_lora` 再 `load_lora`(`:24-26`)。
- 没命中就 `AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)`(`:28`),即 HF 路线。
- 打印参数量用的 `get_model_params` 从公共工具 import(`eval_llm.py:9` → `trainer/trainer_utils.py:20`)。别和 `trainer/trainer_utils.py:121` 那个同名 `init_model` 搞混:那是训练脚本按 `--from_weight` 载权重用的,`eval_llm.py` 用的是自己文件里这一份。
- 收尾统一 `.half().eval().to(device)`(`:30`):转 fp16、切推理模式(关闭 dropout 一类训练态行为)、搬到目标设备。

### 交互与关键分支(`eval_llm.py:51-91`)

问题:既想一键跑几个固定问题,又想手动连着聊。

- 内置 8 个中文 prompt(`:51-60`);启动后输入 `0` 自动测试、`1` 手动输入(`:64`,该模式是死循环,`Ctrl+C` 退出)。
- 流式靠 `TextStreamer`(`:65`),挂在 `model.generate(..., streamer=streamer)` 上(`:82-87`),文字是一段段直接打进终端的;`skip_prompt=True` 表示只打回答不打重复输入。
- 权重分支(`:73-76`):`--weight` 含 `pretrain` 时把输入拼成 `bos_token + 问题` 做**裸续写**(`:73-74`)—— 预训练模型只学过"接着往下写",没学过问答格式;其它权重走 `tokenizer.apply_chat_template(conversation, add_generation_prompt=True, open_thinking=...)`(`:75-76`),套上对话模板当指令模型用。
- 历史裁剪在 `:71`,每轮把回答追加回 conversation(`:89`)以便继续聊;速度打印在 `:91`。

## Tool Call 专项评测:`scripts/eval_toolcall.py`(240 行)

问题:普通问答测不出"模型会不会正确调工具" —— 选错函数、参数写成字符串、漏了标签,这些都得单独看。

- 后端开关 `--backend local|api`(`:204`):`local` 用本地权重自己生成(`init_model` `:57-67`,和 `eval_llm.py` 同套路,但没暴露 LoRA 参数);`api` 用 OpenAI 客户端连一个已起好的服务(`:224`),默认地址 `http://localhost:11434/v1`(`:216`,这是 ollama 的端口),模型名由 `--api_model` 给(`:218`)。
- 内置 8 个工具的 schema(`TOOLS` `:18-27`:数学计算、当前时间、随机数、字数统计、单位换算、天气、汇率、翻译)和它们的**假实现**(`MOCK_RESULTS` `:29-38`):天气、汇率这类直接返回写死的值,数学题走真求值的 `safe_math_eval`(`:30` → `trainer/trainer_utils.py:183`)。只考察模型会不会调、调得对不对,不掺杂外部服务成败。`get_tools()`(`:42-43`)按名字取子集。
- 8 个标准 case 在 `TEST_CASES`(`:45-54`),每条给 prompt + 可用工具子集;自动模式逐条跑,手动模式开放全部工具(`:228`)。
- 解析模型输出:本地模式看的是原始文本,`parse_tool_calls`(`:70-78`)用正则抠 tool_call 标签里的 JSON;API 模式服务端通常已返回结构化 `tool_calls`,若没有,`parse_tool_call_from_text`(`:81-96`)再从正文兜底抠一遍(`:144`、`:173`)。
- 多轮闭环在 `run_case()`(`:177-199`):拿到调用 → `execute_tool()`(`:99-112`)执行 → 结果作为 `role=tool` 消息回喂 → 再生成,直到模型不再调工具(`:185-186`)。本地/API 两种模式的 assistant、tool 消息形状不同(`:192`、`:199`,API 模式要带 `tool_call_id`)。
- **脚本不打分、不判对错**:`run_case` 里没有任何断言,只有把过程打印出来给人看 —— 发起调用 `📞 [Tool Calling]`、执行结果 `✅ [Tool Called]`(`:196-198`)。
- 路径默认值是按"在 `scripts/` 目录下运行"写的:`--load_from ../model`(`:205`)、`--save_dir ../out`(`:206`),所以命令是 `cd scripts && python eval_toolcall.py --weight full_sft`。测本项目自带服务时要显式改 `--api_base_url`,见末尾"新手常见坑"。

## 为什么训好的 `.pth` 还要"转换格式"

问题:`out/full_sft_768.pth` 里只有张量,没有"这些张量属于什么网络结构"的说明。

- 原生 `.pth` 是 PyTorch 的张量集合,读它的人必须**先有本项目的模型代码**(配置类和模型类都在 `model/model_minimind.py`:`MiniMindConfig` 在 `:10`、`MiniMindForCausalLM` 在 `:239`)才知道怎么把张量装回去。vllm、sglang、ollama、llama.cpp 这些通用工具手里没有这段代码,拿到 `.pth` 无从下手。
- HF 格式把这份说明显式打包进目录:`config.json` 写结构参数、权重文件存张量、`tokenizer.json` / `tokenizer_config.json` 存分词器和对话模板;`AutoModelForCausalLM.from_pretrained(目录)` 靠 `config.json` 里的 `auto_map` 找到模型类,所以任何装了 `transformers` 的环境都能加载(这也是前面各脚本 `trust_remote_code=True` 的用途)。目录结构示例见 README 的「🖥️ 基于 MiniMind 的 API 服务接口」一节。
- 于是这条线是固定的:**训练阶段用原生 `.pth`(和 PyTorch 优化器、混合精度、断点续训天然契合),部署阶段转成 HF 目录(换生态兼容性)**。[10 篇](10_rollout_engine.md) 里 sglang 同步权重要 `save_pretrained`(`trainer/rollout_engine.py:183`)也是同一个道理:server 侧只认 HF 目录。
- README 的「🔧 模型转换」一节同样写了这条约定:开源发布以 transformers 格式为主,用原生权重请先转换。

### 转换函数表(`scripts/convert_model.py`,144 行)

| 函数 | 位置 | 用途 |
|---|---|---|
| `convert_torch2transformers_minimind` | `:16-36` | 原生 `.pth` → HF 目录,仍用本项目自己的模型类(先 `register_for_auto_class`,让 Auto 类能找到它,`:17-18`)。注意 `save_pretrained(..., safe_serialization=False)`(`:26`)存的是 `pytorch_model.bin`,不是 safetensors |
| `convert_torch2transformers` | `:40-96` | 通用版:把权重搬进 transformers 官方已有的 `Qwen3ForCausalLM` / `Qwen3MoeForCausalLM` 结构(`:56-71`),不再依赖自定义类,兼容性最好;MoE 时要逐专家的权重合并成 `gate_up_proj`(`:73-79`)。`load_state_dict(..., strict=True)`(`:81`),名字对不上直接报错。默认存 safetensors(`:83`) |
| `convert_transformers2torch` | `:99-102` | 反向:HF 目录 → 原生 `.pth`,用来接手上的 HF 模型继续在本项目里训练 |
| `convert_merge_base_lora` | `:105-112` | 把 LoRA 增量合并回基座、存成新的完整 `.pth`,内部调 `merge_lora`(`model/model_lora.py:56`,import 在 `convert_model.py:12`) |
| `convert_jinja_to_json` / `convert_json_to_jinja` | `:115-118` / `:121-125` | 对话模板在 `.jinja` 文本与 `tokenizer_config.json` 字段之间互转,改模板时用 |

这个脚本**没有 argparse**,参数硬写在 `__main__` 里:`lm_config`(768 维 / 8 层 / 8192 长度 / 非 MoE,`:129`)、输入 `../out/full_sft_768.pth`(`:132`)、输出 `../minimind-3`(`:133`),实际执行的是通用版(`:134`);LoRA 合并、反向转换等备用调用以注释形式留在 `:136-144`,要用就取消注释。所有路径都以 `../` 开头,所以必须 `cd scripts && python convert_model.py`。

## OpenAI 兼容服务:`scripts/serve_openai_api.py`(252 行)

问题:别人想在自己的聊天前端里用你的模型,得有个标准接口,而不是让人去抄你的 Python。

- Web 框架是 FastAPI(`:16`,用 `FastAPI()` 建 app、用装饰器把函数挂到某个 URL 上);真正对外服务的是 uvicorn,启动时先加载模型再起服务(`:250-252`)。
- `init_model`(`:28-47`)与 CLI 一样两条路线都支持:原生 `.pth` 路径拼法是 `../{save_dir}/{weight}_{hidden_size}{_moe}.pth`(`:32`,以 `scripts/` 为基准),也支持 HF 目录(`:45`)和挂 LoRA(`:41-43`)。
- **只有一个业务路由**:`POST /v1/chat/completions`(`:178-234`)。`/v1/chat/completions` 这个路径名和请求/返回字段就是 OpenAI 定义的那套,所以叫"兼容"—— 前端软件只要把服务地址指到这里,就能不改代码地换成你自己的模型。
- 请求体 `ChatRequest`(`:50-68`):标准字段 `model`/`messages`/`temperature`(`0.7`)/`top_p`(`0.92`)/`max_tokens`(`8192`)/`stream`(默认 True,`:51-56`);扩展字段 `tools`(`:57`,工具清单)、`open_thinking`(`:58`,是否显式思考)、`chat_template_kwargs`(`:59`,`get_open_thinking()` `:61-68` 把这两种写法都认,因为不同前端习惯不同)。
- 启动命令 `cd scripts && python serve_openai_api.py`,监听 `0.0.0.0:8998`(`:252`,地址与端口都写死)。`0.0.0.0` 意思是"接受来自任何网卡的连接",所以同网段机器也能访问(暴露到公网前要想清楚鉴权)。
- `stream=true` 走 `generate_stream_response`(`:105-175`):生成放在**子线程**里跑(`:113-130`),主循环从队列一段段取文字、拼成 SSE 事件(即 `data: {...}` 一行行推给客户端的流式协议)返回(`:136-172`),外层用 `StreamingResponse`(`:182-192`)。跨线程传话靠 `CustomStreamer`(`:71-80`,继承 HF 的 `TextStreamer`,每产出一段就 `queue.put`,结束时 put 一个 `None` 收尾,`:77-80`)。为什么要另起线程:生成是阻塞的死活儿,不分开就没法边生成边往网络里写。
- `stream=false` 就一次性生成、解码,返回标准 `chat.completion` JSON(`:193-232`)。
- `parse_response`(`:83-102`)是"能不能被现成前端直接用"的关键:把 think 起始/结束标签之间的内容摘出来放进 `reasoning_content` 字段(`:85-92`,还容忍模型漏写起始标签的情况),把 tool_call 标签里的 JSON 转成标准 `tool_calls` 结构(id / type / function.name / function.arguments,`:93-101`)并从正文里删掉(`:101`)。
- 流式路径按同样的字段下发:遇到思考结束标签之前的片段走 `delta.reasoning_content`,之后走 `delta.content`(`:145-167`),结尾统一补 `tool_calls` 与 `finish_reason`(`:169-172`,有工具调用时 `finish_reason` 为 `tool_calls`)。前端因此能把思考过程折叠显示、把工具调用交给真执行器。
- 自测客户端:`scripts/chat_api.py`(46 行)是最小 OpenAI 脚本对话循环,`base_url` 写死 `http://localhost:11434/v1`(`:3-6`),并通过 `extra_body.chat_template_kwargs.open_thinking` 传思考开关(`:21`)—— 正好用来验证 `reasoning_content` 有没有真的透传出来。

## Streamlit WebUI:`scripts/web_demo.py`(424 行)

问题:给不写代码的人一个能打字、能看见思考和工具调用的界面。

- 启动:`cd scripts && streamlit run web_demo.py`(README 的「3'（可选）WebUI」一节)。Streamlit 的机制是从上到下把整个脚本当页面跑一遍、每次交互都重跑,靠 `@st.cache_resource`(`:202`)让模型只真正加载一次(`load_model_tokenizer` `:202-213`)。
- **只走 HF 路线**:`AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)` 加同名 tokenizer(`:204-211`),读不了 `.pth`。
- 模型从哪来:启动时扫描**脚本自己所在目录**的子文件夹,凡含 `.bin` / `.safetensors` / `.pt` 或分片索引文件的就列进侧栏下拉框(`:244-252`);一个都没扫到就显示占位项 `No models found`(`:252`)。这就是 README 强调"必须先把 transformers 格式的模型文件夹复制到 `./scripts/` 下"的原因。
- 侧栏控件:语言(`:262-268`)、历史轮数 / 最大生成长度 / 温度三个滑条(`:273-275`)、思考开关 `enable_thinking`(`:280`)、工具勾选(最多 4 个,`:282-290`,工具表 `TOOLS` `:111-120`)。
- 页面内置的工具是**假的**:`execute_tool`(`:128-150`)里天气返回固定"晴, 7~10°C"、汇率固定 7.2、翻译固定 "hello world" 类字样,只有数学计算走真求值 `safe_math_eval`(`:132`)。目的是演示工具链路,不是提供真功能。
- 生成主流程在 `main()`(`:316-420`):按侧栏设置裁剪历史、不带工具时补一条中文 system 提示(`:355-356`)→ 按开关决定给 `apply_chat_template` 传 `open_thinking` / `tools`(`:357-362`)→ 用 `TextIteratorStreamer`(`:366`)在子线程里生成(`:367-380`),主线程每收到一段就重渲染气泡(`:383-385`)。这就是网页版流式输出。
- 工具调用闭环在 `:388-417`:最多 16 轮,发现 tool_call 标签就执行、把结果当 `role=tool` 消息回喂、再重新生成(`:397` 调 `execute_tool`,`:404-411` 重新拼模板并再生成)。
- `process_assistant_content`(`:153-199`)负责把标签变成可读界面:tool_call 渲染成带背景色的卡片(`:155-164`),思考内容放进可折叠区块,还在思考没结束时提前折叠(`:167-175`),并兼容三种残缺情况:标签成对(`:177-183`)、只有起始(`:185-189`)、只有结束(`:191-197`)—— 流式输出中途必然出现半截标签,不处理就会把原始标签暴露给用户。
- `regenerate_answer`(`:237-240`)弹出最后一轮 assistant 消息后 `st.rerun()`,让这一轮重答(`:335-340` 负责接住这个重答标记)。
- 脚本主体在 import 阶段就建好 `MODEL_PATHS`、渲染侧栏(`:244-313`),所以**改代码后直接重启 streamlit 进程**才能生效。

## 第三方推理引擎

前提都是先有 HF 格式目录(上一节)。两条转换路线决定了第三方工具看到的是本项目自定义模型类(`convert_torch2transformers_minimind`,靠 `trust_remote_code` 找到它)还是 transformers 官方的标准 Qwen3 类(`convert_torch2transformers`)。具体命令都写在根 `README.md` 里,按章节标题找比按行号找稳:

| 工具 | 是什么 | 命令所在章节 |
|---|---|---|
| sglang | 高性能推理引擎(README 描述其支持 RadixAttention、连续批处理),也是 [10 篇](10_rollout_engine.md) 的 rollout 引擎 | README「其他」下的 SGLang 小节:`python -m sglang.launch_server --model-path ... --attention-backend triton --host 0.0.0.0 --port 8998` |
| vllm | 目前非常常用的高效推理框架 | README「其他」下的 vllm 小节:`vllm serve /path/to/model --model-impl transformers --served-model-name minimind --port 8998` |
| ollama | 本地运行大模型的常用工具,吃 GGUF | README「其他」下的 ollama 小节:先用 llama.cpp 转 GGUF,再在模型目录写 `minimind.modelfile` 导入 |
| llama.cpp | 轻量 C++ 推理框架,支持多线程与部分 GPU 加速 | README「其他」下的 llama.cpp 小节:改 `convert_hf_to_gguf.py` 后转 GGUF,再用 `llama-quantize` / `llama-cli` |
| MNN | 面向端侧(手机 / Mac)的推理引擎 | README「其他」下的 MNN 小节:`llmexport.py` 导出、`llm_demo` 测试 |
| 只想先跑起来试试 | 最省事的两条命令 | README「Ⅰ 🚀 模型推理」的「4'（可选）第三方推理框架」一节:`ollama run jingyaogong/minimind-3`、`vllm serve /path/to/model --served-model-name minimind` |

## 客观评测(选择题基准)

问题:上面这些都是"看一眼感觉",想要可比较的数字得跑公开数据集。

用第三方框架 `lm-evaluation-harness`(命令与结果表都在 README 的「Ⅴ 客观评测」一节,含安装步骤与数据集清单 C-Eval / CMMLU / ARC-Easy / PIQA / OpenBookQA / HellaSwag / Social-IQa)。要点照 README:指令模型评测要加 `--apply_chat_template`,纯基座模型不加;跑 HF 目录路径。README 同一节也提醒:这类微型模型的分数常年贴着"随机猜"的下界,差几个百分点不代表能力差异,结果仅供参考。

## 本机现状与可跑命令(用 `ls` 确认过,本次改写未实际执行推理)

- `out/` 下现有 `pretrain_768.pth`、`full_sft_768.pth` 两个原生权重;`checkpoints/` 下是对应的 `_resume.pth` 断点状态文件,**不是**用来推理的权重。
- 仓库根目录**没有** `minimind-3` 这种 HF 目录,`scripts/` 下也没有模型子目录,因此 `web_demo.py` 现在只会显示 `No models found`。
- 现在就能跑:

```bash
python eval_llm.py --weight full_sft --max_new_tokens 256      # 启动后输入 0 走内置自动测试
python eval_llm.py --weight pretrain --max_new_tokens 256      # 裸续写,输出不会是问答格式,属正常
cd scripts && python eval_toolcall.py --weight full_sft        # 工具调用评测(需要 out/full_sft_768.pth)
cd scripts && python serve_openai_api.py --weight full_sft     # 起在 8998 端口
```

- 需要先准备权重或模型目录才能跑:`--weight grpo|agent|dpo|...`(先按 [05](05_dpo.md)~[08](08_agentic_rl.md) 篇训练,或按 README 从 ModelScope/HuggingFace 下载,见 README 的「2.3 测试已训练模型（可选）」一节);`web_demo.py` 与第三方引擎都要 HF 目录,先 `cd scripts && python convert_model.py` 转出 `../minimind-3`,网页版再 `cp -r minimind-3 ./scripts/minimind-3`。

## 新手常见坑

1. **`--load_from` 是"字符串里含 `model`"的判定,不是严格枚举**(`eval_llm.py:14`、`eval_toolcall.py:59`、`serve_openai_api.py:30`)。如果你的 HF 目录路径里正好带 `model` 这个词(比如 `./my_test_model`),脚本会以为要用原生 `.pth`,转而去找 `out/{weight}_{hidden_size}.pth`,报的是文件不存在,看着很莫名其妙。
2. **默认服务地址对不上**。`serve_openai_api.py` 起在 8998(`:252`),但 `chat_api.py:5` 和 `eval_toolcall.py:216` 的默认 `base_url` 都写着 11434(ollama 的端口)。用 `--backend api` 测本项目服务要显式加 `--api_base_url http://localhost:8998/v1`;另外 8998 同时是 sglang 的默认端口([10 篇](10_rollout_engine.md)),两者不能同时起。
3. **`scripts/` 下的脚本必须在 `scripts/` 里运行**。权重路径直接以 `../` 拼(`serve_openai_api.py:32`、`eval_toolcall.py:205-206`、`convert_model.py:132`),在仓库根跑就会去找仓库外面不存在的 `../out`。反过来 `eval_llm.py` 用的是 `./out`(`:22`,默认 `--save_dir out`),必须在仓库根跑。
4. **规模参数要和训练时一致**。原生路线是 `strict=True` 加载(`eval_llm.py:23`),`--hidden_size` / `--num_hidden_layers` 填错会直接报参数缺失或形状不匹配;MoE 权重还得加 `--use_moe 1`,否则文件名后缀 `_moe` 拼不出来(`:21-22`)。另外 `--max_new_tokens` 默认 8192,冒烟测试建议调小,help 里也写了这个值不代表模型真实长文能力(`:42`)。

## 学完之后能做什么

到这里 14 篇链路就走完了:你已经能从分词器一路训出权重,在终端里验证它、给它套上对话模板和工具调用能力,再把权重转成通用格式对外提供 API 或网页服务。
接下来常见的三条路:换更大的 `--hidden_size` / 加 MoE 重跑 [02](02_pretrain.md)、[03](03_full_sft.md) 两条主线;挑一条增强路线深挖([04 LoRA](04_lora.md)、[05 DPO](05_dpo.md)、[06 PPO](06_rlaif_ppo.md)、[07 GRPO/CISPO](07_grpo_cispo.md)、[08 Agentic RL](08_agentic_rl.md)、[09 蒸馏](09_distillation.md));或者把 `serve_openai_api.py` 接进自己的应用。全流程索引与公共训练基建见 [文档首页](README.md)。
