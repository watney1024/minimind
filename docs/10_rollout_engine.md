# 10 Rollout 引擎(`trainer/rollout_engine.py`,224 行)

## 这一步在干什么(人话)

强化学习训练每一步要做两件事:先让模型自己把回答"写"出来,再根据写出来的东西好不好去调整模型参数。前半件事叫 rollout(展开采样),它不是训练,只是推理,但恰恰是整轮训练里最花时间的那部分。
本项目把"生成回答"单独抽成一个可以替换的引擎:默认用 PyTorch 自己手写的一步步生成,也可以换成专门的推理服务 sglang 来批量生成,训练脚本(`train_ppo.py` / `train_grpo.py` / `train_agent.py`)一行都不用改。
换引擎有一个必须处理的代价:负责生成的那一侧也存着一份模型权重,模型每更新一次,就得把新权重同步过去 —— 这就是接口里 `update_policy` 存在的原因。

## 先懂这几个词

| 术语 | 大白话 |
|---|---|
| 推理 / 生成(inference / generate) | 拿模型一个字一个字往外输出答案的过程,不改模型参数 |
| rollout | RL 里的说法,指"用当前模型把一批 prompt 的回答生成出来",本质就是推理 |
| token | 模型处理文字的最小单位,本项目由分词器切出(见 [00 篇](00_tokenizer_dataset.md)) |
| logprob | 模型输出某个 token 时对它的对数概率,取对数后的"我有几成把握"。数值越大越自信 |
| old logprob | 采样那一刻记录的 logprob,当作后续算比率的"基准值" |
| 重要性采样 / ratio | 权重更新后同一个 token 的 logprob 会变大变小,新旧相除(`exp(新-旧)`)就是比率;RL 的 loss 靠它判断"这次更新迈了多大步" |
| 组采样 / `num_generations` | 同一个 prompt 复制成多份、一次生成多条不同回答,GRPO 靠这一组回答互相比优劣 |
| server / HTTP API | 一个常驻的网络服务,别的程序通过发请求(如 `POST /generate`)让它干活,不必在同一个进程里 |
| HF 格式 / transformers 目录 | HuggingFace 那套"一个文件夹 = config.json + 权重 + 分词器 + 对话模板"的通用模型存放格式,几乎所有第三方工具都认它 |
| DDP / unwrap | 多卡训练时模型被包了一层分布式外壳,想直接调它的方法要先剥掉外壳 |
| autocast | 混合精度上下文:同一份代码里自动用 fp16/bf16 的低精度算,换速度 |

## 为什么要把"生成"单独抽出来

这是本篇最重要的一条逻辑,理解了它,代码就只是接口翻译。

- **生成占 RL 单步耗时的大头**。一个 step 里,采样要一条条把回答写完(可能几百个 token),而反向传播只做一次。MiniMind 用的是项目自己手写的 `generate`(`model/model_minimind.py:262`,`MiniMindForCausalLM` 继承 `PreTrainedModel, GenerationMixin`,见 `model/model_minimind.py:239`),循环里逐步前向,不做连续批处理这类专门优化,吞吐有限。
- **专业推理引擎天生就是为了"同时服务很多请求、尽可能快地吐字"而写的**,同一件事能快好几倍。sglang 就是这样一个引擎,它作为独立进程跑,对外提供 HTTP 接口。
- **但两者代码完全不同**(本地 `model.generate()` vs 发 HTTP 请求),如果直接把调用写进训练循环,训练脚本就会被绑死在某种实现上。
- 于是抽成一个统一接口的两个方法:`rollout(...)` 负责"给我 prompt,返回一批回答和它们的 old logprob";`update_policy(model)` 负责"我训练出新权重了,同步给你"。训练脚本只认这两个方法,用哪种引擎由命令行一个开关决定。
- `update_policy` 存在的必要性:sglang server 里的权重是它自己那份,训练侧更新完它并不知道。如果不同步,模型越训越好但 server 还在用旧权重采样,采样分布和被优化的策略彻底脱节,RL 的数学前提就没了。torch 引擎不存在这个问题,所以它的 `update_policy` 只是把引用换一下(几乎 no-op)。

需要说明:当前实现是**同步**模式 —— 采样完一批才开始这一批的更新,不是把 rollout 丢进 buffer 的异步训练(README 的 Agentic RL 一节对此有说明)。

## 统一接口

### `compute_per_token_logps()`(`trainer/rollout_engine.py:24-36`)

问题:RL 的 loss 需要知道"每个 token 当时的概率",但 `generate` 只给你文字,不给你逐 token 的 logprob。

- 做法:把"prompt + 生成的回答"整条序列再前向一次,`log_softmax` 后用 `torch.gather` 只取出实际出现的那个 token 的 logprob(`:33-35`)。
- 省显存:只保留最后 `n_keep + 1` 个位置的 logits(`:29`,靠模型的 `logits_to_keep` 参数),prompt 部分的 logits 直接不算。
- 多卡安全:先 unwrap DDP 外壳再调用(`:27`)。
- 为什么"保证重要性采样自洽":训练侧算新 logprob 用的是同一个模型、同一套前向(`trainer/train_grpo.py:101`),而 old logprob 来自这里(`trainer/train_grpo.py:90` 取 `rollout_result.per_token_logps`)。两者定义完全一致,所以比率 `ratio = torch.exp(per_token_logps - old_per_token_logps)`(`trainer/train_grpo.py:134`)在没有参数更新时恰好≈1;old logprob 也确实来自"生成这条回答的那个策略"。这条自洽性一旦破掉(比如 old 来自另一个 tokenizer 口径、另一套 padding 规则),整轮 RL 的梯度就偏了。
- Agentic RL 里也复用这个函数算参考模型的 logprob:`trainer/train_agent.py:27` 直接 import,`trainer/train_agent.py:280` 调用。

### `RolloutResult`(`trainer/rollout_engine.py:40-47`)

问题:两种引擎返回的东西长相差很远,得有一个统一的数据结构。字段:

| 字段 | 含义 |
|---|---|
| `output_ids` | 完整序列 = prompt + 回答的 token id 张量 |
| `completion_ids` | 只有回答部分的 token id |
| `per_token_logps` | 回答部分逐 token 的 old logprob |
| `completions` | 回答的文本(解码后),给奖励模型打分用 |
| `prompt_lens` | 每条样本 prompt 的长度,用来定位回答从哪一列开始 |
| `completion_mask` | 哪些列是真 token、哪些是补齐用的 padding |

### `RolloutEngine`(抽象基类,`trainer/rollout_engine.py:51-60`)

只规定两件事,子类必须实现:`rollout(prompt_ids, attention_mask, num_generations, max_new_tokens, temperature)`(`:55`)和 `update_policy(model)`(`:59-60`)。

### 工厂 `create_rollout_engine()`(`trainer/rollout_engine.py:209-224`)

问题:训练脚本不想写 if-else 决定用哪种引擎。

- `engine_type == "torch"` → `TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)`(`:219-220`)
- `engine_type == "sglang"` → `SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)`(`:221-222`)
- 其它值直接 `ValueError`(`:223-224`)
- 三个训练脚本都注册了同一组开关:`--rollout_engine torch|sglang`、`--sglang_base_url`、`--sglang_model_path`、`--sglang_shared_path`(`trainer/train_grpo.py:233-236`、`trainer/train_ppo.py:346-349`、`trainer/train_agent.py:402-405`),默认值分别是 `torch`、`http://localhost:8998`、`../model` 和一个本地缓存目录(`./sglang_ckpt_grpo` / `_ppo` / `_agent`)。

## TorchRolloutEngine(`trainer/rollout_engine.py:64-95`)

问题:零外部依赖的默认路径 —— 什么都不用装,单机单卡就能跑起来。

- `rollout`(`:71-92`):在 `torch.no_grad()` 和训练脚本传进来的 autocast 上下文里(`:73-74`)调用本地手写 `generate`(`:75-84`,对应 `model/model_minimind.py:262`)。
- **组采样怎么实现的**:不是靠 `num_return_sequences`,而是先把每条 prompt 复制 `num_generations` 遍 —— `prompt_ids.repeat_interleave(num_generations, dim=0)`(`:76-77`),`generate` 时固定 `num_return_sequences=1`(`:81`)。这样 batch 维度变成 `B*num_gen`,同一个 prompt 的多条副本挨着排列,正好对上 GRPO 按组算均值的 `view(-1, num_generations)`(`trainer/train_grpo.py:121`)。
- 采样完切出回答部分(`:85-86`),拼出完整 attention_mask(`:87`),再用上面那个 `compute_per_token_logps` 补算 old logprob(`:88`),最后解码文本(`:89`)、组装 `RolloutResult`(`:90-92`)。
- `update_policy`(`:94-95`):只把 `self.policy_model` 指向传入的模型。因为采样和训练用的是同一个对象(甚至就是刚被 optimizer 改过参数的那份),所以这一步几乎什么都没做 —— 这也说明 torch 模式下不存在权重同步问题。
- 优点:无外部服务、无端口、无落盘,调试直观。缺点:采样和训练抢同一张 GPU 的显存与算力,64M 的小模型也一样慢,因为逐 token 前向的次数就是那么多。

## SGLangRolloutEngine(`trainer/rollout_engine.py:99-205`)

问题:把最花时间的生成活儿外包给专用引擎,训练卡只负责更新参数(所谓"训推分离")。

- `__init__`(`:100-105`):记住 server 地址(`:101`)、权重落盘目录(`:102`)、超时(`:103`);`sglang_model_path` **只用来加载 tokenizer**(`:104`),所以它指向 `../model` 这种带分词器文件的目录就够了,不需要是完整模型。
- `rollout`(`:107-173`):
  - 先按 attention_mask 去掉每条 prompt 的 padding,只留有效 token(`:109-112`),再把每条 prompt 复制 `num_generations` 遍组成一个批量请求(`:113`)—— 和 torch 版一样是"复制 prompt",而不是让 server 内部出 `n` 条。
  - 组 payload:`sampling_params` 带 `temperature` / `max_new_tokens` / `stop_token_ids`(用 eos token,`:115-123`),`"return_logprob": True` 要求把生成 token 的 logprob 一起回传(`:122`)。
  - `POST {base_url}/generate`(`:125`),`raise_for_status()` 让非 2xx 直接抛错(`:126`)。
  - 从返回的 `meta_info.output_token_logprobs` 里取 old logprob(`:136-145`);条数与生成 token 数不一致时,前面补 `0.0` 或从尾部截断对齐(`:147-150`),避免长度错位把 loss 算歪。
  - 变长序列用 `pad_to_tensor` 补齐成张量(`:158-168`),padding 位置的 logprob 填 `0.0`(`:169`),同时给出 `prompt_lens`(`:171`)和标记真 token 的 `completion_mask`(`:172`)—— loss 侧必须靠这个 mask 屏蔽补齐位(用法见 `trainer/train_grpo.py:126-130`)。
- `update_policy`(`:175-194`):这是"代价"所在,四步:
  1. 只在主进程做(`:177`),unwrap DDP、再去掉可能的 `_orig_mod` 包装(`:179-180`);
  2. 权重转 fp16 搬回 CPU(`:182`),用 `save_pretrained(..., safe_serialization=False)` 存成 HF 目录(`:183`),顺带存 tokenizer(`:184`);
  3. `POST {base_url}/update_weights_from_disk`,把刚写的目录路径告诉 server,让它重新加载权重(`:185-187`);
  4. 多卡下把这个成功/失败标志广播给所有 rank 并 barrier 同步(`:190-192`),失败就抛 `RuntimeError`,宁可训练崩也不要悄悄用旧权重继续采样(`:193`)。
- `flush_cache`(`:196-198`):`POST /flush_cache`,让 server 丢弃内部缓存。**当前代码里训练脚本没有调用它**,它只是备好的方法。
- `health`(`:200-205`):`GET /health` 探活,异常返回 False。同样,**训练循环里没有调用**,server 挂了要到 rollout 发请求时才报错。
- 收益与代价:训练与推理在不同进程/不同 GPU 上,采样吞吐明显提高;代价是每次同步要写一次盘、server 重载一次权重,以及必须自己保证 server 起着、模型目录是 HF 格式。

## 使用

先起 sglang server(它吃的是 transformers 格式目录,转换方法见 [11 篇](11_eval_deploy.md)):

```bash
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
```

这条命令就写在 `trainer/rollout_engine.py:1-4` 的文件头注释里。然后训练侧(在 `trainer/` 目录下执行,因为 `--sglang_model_path` 默认是 `../model`):

```bash
# 默认 torch,不依赖任何外部服务
python train_grpo.py

# 用 sglang 加速 rollout
python train_agent.py --rollout_engine sglang --sglang_base_url http://localhost:8998 --sglang_shared_path ./ckpt_mm
```

第二个例子取自 README 的 Agentic RL 一节(可用 `--data_path ../dataset/agent_rl_math.jsonl` 等参数,详见 [08 篇](08_agentic_rl.md))。PPO、GRPO 用同一组开关,见 [06 篇](06_rlaif_ppo.md)、[07 篇](07_grpo_cispo.md)。

## 新手常见坑

1. **以为选了 sglang 就自动有 server**。这个引擎只是客户端,server 不会自己起,也不会自动帮你转格式;而且 `health()` 虽然实现了(`:200-205`)却没在训练循环里被调用,所以 server 没起/挂了,报错点在第一次 `rollout` 的请求上(`:125-126`)。
2. **8998 端口撞车**。sglang 的默认端口和训练脚本的 `--sglang_base_url` 默认值都是 8998(`:3`、`trainer/train_grpo.py:234`),而 `scripts/serve_openai_api.py:252` 也把这个端口写死了。两者不能同时起。
3. **权重同步频率绑在存档频率上**。`update_policy` 只在 `step % args.save_interval == 0` 时被调用(`trainer/train_grpo.py:193`、`trainer/train_ppo.py:256`、`trainer/train_agent.py:361`)。两次同步之间 sglang 用的还是旧权重,采出来的回答对新参数来说是 off-policy —— 这是实现上故意的取舍,靠比率截断兜住,不是 bug;但把 `save_interval` 调得很大时采样会明显滞后。
4. **别以为 padding 位的 logprob 是真值**。sglang 分支里补齐位置填的是 `0.0`(`:169`,等价于"概率 1"),真实含义靠 `completion_mask`(`:172`)屏蔽。写新引擎时忘了给 mask,loss 会把 padding 当成高置信 token 学。

## 下一步

引擎换成 sglang 只是让 RL 训练更快,训完的模型怎么测、怎么起服务在 [11 评估与部署](11_eval_deploy.md)。
