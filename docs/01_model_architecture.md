# 01 模型结构(`model/model_minimind.py`,293 行,末行无换行符)

一句话:这一步定义了"模型长什么样"——一个手写、对齐 Qwen3 生态的 Decoder-only Transformer。

## 这一步在干什么(人话)

模型结构是所有训练脚本共用的骨架:预训练、SFT、LoRA、DPO、各种 RL,换的是"用什么数据、算什么 loss",不换的是"这套积木怎么拼"。它不在训练链路的某一段里,而是躺在所有段下面——你完全可以不改它就跑完整条流程。但一旦想知道"训练到底在动哪些参数""为什么生成第 100 个字不用重读前 99 个字",就必须读它。本篇只讲结构和两个关键机制,不讲怎么训它。

## 先懂这几个词

| 术语 | 大白话 |
|---|---|
| token | 文本被切成的最小单位,可以是一个字、一个词或半个词;模型只认编号,不认字符串 |
| embedding(词嵌入) | 把每个 token 编号换成的一串数字(向量),让意思相近的词在向量空间里也靠得近 |
| 层 / block | 模型重复堆叠的同一种加工单元;本模型默认堆 8 层(`:12`) |
| MLP(前馈网络) | 每层里的"全连接加工站":先放大到更宽的空间(768→2432)再压回去,做逐 token 的非线性变换 |
| 残差 | 把支路算出来的东西**加**回原输入(`x + f(x)`)而不是**替换**,这样信息不会被逐层冲掉,梯度也更好传 |
| attention head(注意力头) | 模型同时从几个不同角度问"该看前面哪些 token",每个角度就是一个头;默认 8 个头(`:22`) |
| GQA(分组查询注意力) | 多个查询头共用一份 key/value,让缓存和显存按头的比例省下来;本模型 8 个 Q 头共用 4 份 KV(`:22-23`) |
| KV cache | 推理时把历史 token 的 key/value 存起来,下一步只算新 token,不必重读全文(详见下文主线二) |
| RoPE(旋转位置编码) | 不额外加"位置向量",而是按 token 在句中的位置把 q/k 向量旋转一个角度,让注意力天然知道先后次序 |
| RMSNorm / 归一化 | 把每个 token 的向量按它自己的均方根长度缩放到统一量级,防止数值一路放大或塌缩;RMSNorm 是 LayerNorm 的简化版,不减均值、只有可学缩放系数 |
| SwiGLU | 一种 MLP 写法:两路投影,一路过 SiLU 激活、另一路当"门"逐元素相乘后再降维 |
| MoE / router / 专家 | 混合专家:每层放好几个并列的 MLP(专家),由一个"分配员"(router)给每个 token 挑该进哪个专家,参数变多但每次只算一小部分 |
| logits | 模型最后一个向量对词表每个 token 的打分(未归一化),形状是 (位置数, vocab_size);要变成概率还得 softmax |
| tie_word_embeddings | 输入 embedding 和输出投影 `lm_head` 共用同一份权重矩阵,省掉一整个 vocab_size×hidden_size 的参数 |
| temperature / top_p / top_k | 采样时的三个旋钮:temperature 调分布软硬(越大越敢冒险),top_k 只留分数最高的 k 个候选,top_p 只留累计概率到 p 的候选 |

## 配置 `MiniMindConfig`(`model_minimind.py:10-45`)

配置存在的意义:所有训练脚本只传 `hidden_size` / `num_hidden_layers` / `use_moe` 三个参数(如 `trainer/train_pretrain.py:108`),其余几十个维度都由这里的默认值兜住,保证训练和推理建出来的模型一模一样。它继承 `PretrainedConfig`(`:10`),因此能被 transformers 生态直接读写。

- 骨架尺寸:`hidden_size=768`、`num_hidden_layers=8`、`use_moe=False`(`:12`,每个 token 用多长的向量表示、堆几层、要不要 MoE)。
- `vocab_size=6400`(`:18`,词表大小要和 [00 分词器](00_tokenizer_dataset.md) 训出的 tokenizer 对齐,否则编号对不上)。`bos_token_id=1`、`eos_token_id=2`(`:19-20`)。
- 注意力宽度:`num_attention_heads=8`(`:22`)、`num_key_value_heads=4`(`:23`,GQA:8 个查询头只配 4 份 KV,`n_rep = 8 // 4 = 2`,`:97`)、`head_dim = hidden_size // num_attention_heads`(`:24`,默认 768/8=96)。头数是"多想几个角度"的代价,KV 头数减半是"少存一半缓存"的省钱手段。
- `intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64`(`:26`)。这个数要解决的是"MLP 该开多宽":代码里就是隐藏维度乘 π、再向上取整到 64 的倍数(默认 768 → 2432),项目里当作趣味设计。`hidden_act='silu'`(`:25`)。
- `max_position_embeddings=32768`(`:27`,RoPE 角度表预计算到多远)、`rms_norm_eps=1e-6`(`:28`,归一化分母里的防零小量)、`rope_theta=1e6`(`:29`,旋转频率的基底,越大越能区分长距离位置)。
- `tie_word_embeddings=True`(`:30`):embedding 与 `lm_head` 共享权重(`:247` 直接把两个模块的权重指向同一个张量,实测 `lm_head.weight is model.embed_tokens.weight` 为 True)。
- YaRN 长文本外推:`inference_rope_scaling`(`:31`)为真时给出 `rope_scaling` 参数字典(`:32-39`,`factor=16`、`original_max_position_embeddings=2048`),否则为 `None`。它解决"训练只见过 2k 长,推理想跑更长"的问题——把低频位置分量压缩,高频分量基本不动。
- MoE 字段(`:41-45`,仅 `use_moe=True` 生效):`num_experts=4`(`:41`)、`num_experts_per_tok=1`(`:42`,每个 token 只走 1 个专家)、`moe_intermediate_size` 默认等于 `intermediate_size`(`:43`)、`norm_topk_prob=True`(`:44`)、`router_aux_loss_coef=5e-4`(`:45`,均衡 loss 的权重)。

## 主线一:一层 block 前向一次,到底算了什么

`MiniMindBlock`(`model_minimind.py:183-199`)是重复 8 次的那个加工单元。它的 `forward`(`:191-199`)只有两条支路,都是"归一化 → 算 → 加回原值"的 pre-norm 残差结构:

1. 先记住原始输入 `residual = hidden_states`(`:192`)。
2. 注意力支路:把 `input_layernorm(hidden_states)`(`:194`)送进 `self_attn`(`:193-196`),结果 `+= residual`(`:197`)。归一化(`:187`)是为了让进注意力的向量量级稳定;残差相加是为了"注意力只是在原文上补一层上下文信息",而不是把原文覆盖掉。
3. MLP 支路:`hidden_states + mlp(post_attention_layernorm(hidden_states))`(`:198`)。归一化(`:188`)后再放大-激活-压缩,负责逐 token 的非线性加工。

注意力内部(`Attention.forward`,`:111-134`),顺序值得记牢:

- `:113` 三个矩阵投影出 q/k/v:`q_proj` 是"我要问什么",`k_proj` 是"我这儿有什么标签可供匹配",`v_proj` 是"匹配上之后实际要取的内容"。三者都 `bias=False`(`:100-103`)。
- `:114-116` 把一长条向量拆成 (batch, 序列, 头, head_dim);注意 q 是 8 头、k/v 只有 4 头。
- `:117` q、k 各过一次头级 RMSNorm(即 QK-Norm,模块定义在 `:104-105`,维度是 `head_dim`)。它治的是"个别头数值过大导致注意力饱和"。
- `:118-119` 加 RoPE(`apply_rotary_pos_emb` `:80-84`,内部 `rotate_half` `:81` 把向量后半取负与前半交换后与前半配对,构成二维旋转)。角度表在 `precompute_freqs_cis`(`:62-78`)里预计算成 `cos/sin` 两张表,`freqs_cos`/`freqs_sin` 形状为 (32768, 96)(实测)。
- `:120-122` 如果有缓存就先把历史 k/v 拼在自己前面(主线二)。
- `:124` `repeat_kv`(`:86-89`)把 4 份 KV 各复制 `n_rep=2` 份凑成 8 头,并把维度转成 (batch, 头, 序列, head_dim)。实测复制顺序是 `[kv0,kv0,kv1,kv1,kv2,kv2,kv3,kv3]`,正好让第 i 个 Q 头对上第 `i // 2` 份 KV。
- `:125-126` 满足条件时走 PyTorch 的 `F.scaled_dot_product_attention`(即 flash attention 路径,是否启用由 `:109` 的 `self.flash` 决定);否则 `:128-131` 手写 `softmax(QKᵀ/√head_dim + 因果遮罩) @ V`,因果遮罩在 `:129`(上三角填 `-inf`,让自己看不见未来的 token),padding 遮罩在 `:130`。
- `:132-133` 多头拼回一长条,再过 `o_proj` 混合头的信息。

MLP 有两种:

- `FeedForward`(`:136-146`):`down(act(gate(x)) * up(x))`(`:146`),这就是 SwiGLU。`gate_proj` 负责"要不要放行"(`:140`),`up_proj` 负责内容(`:142`),`down_proj` 压回 768(`:141`)。为什么要有它:注意力只负责"看谁",真正存知识的稠密计算在这两路乘法里。
- `MOEFeedForward`(`:148-181`):`gate`(`:152`)是 router,对每个 token 出一个 4 维 softmax 分数,`topk`(`:160`)挑 `num_experts_per_tok` 个专家(`:153` 建了 4 个 `FeedForward`)。`k>1` 时把选中的权重归一(`:161-163`);`k==1`(默认)时把前向权重强制设为 1.0、只让梯度直传(`:165-166` 的 `top1 - top1.detach() + 1.0`,即 straight-through)。`:167-173` 逐专家收集属于自己的 token 算完再 `index_add_` 累加;`:174-175` 给本轮没被选中的专家也挂一个 0 系数,好让它们在反向图里不消失。`aux_loss` 在 `:176-180` 算:选中频率 `load` 与平均门控分数相乘再乘 `num_experts * router_aux_loss_coef`(`:178`),它治的是"router 偏心,只有少数专家被反复训练"这个病;推理时置零(`:180`)。

外层 `MiniMindModel`(`:201-237`)把这些拼起来:`embed_tokens`(`:206`)把编号变向量 → dropout → 8 个 block 串着过(`:208`、`:226-234`)→ 最后 `norm`(`:209`、`:235`),并把所有 MoE 的 `aux_loss` 求和(属性,见 `:236`)。

最外圈 `MiniMindForCausalLM`(`:239-293`)只做两件事:加 `lm_head`(`:246`)把向量投影回词表得到 logits,以及可选地算 loss(`:255-257`)。它继承 `PreTrainedModel` 和 `GenerationMixin`(`:239`)。

## 主线二:KV cache 为什么能加速

问题从"自回归生成"来:模型一次只吐一个 token,新 token 又要进序列参与下一个 token 的计算。naive 做法每生成 1 个字就把整段前缀重新前向一遍,而前缀里每个 token 的 k/v 其实和上一步算的一模一样——纯属重复劳动。

代码里的解法(`:120-123`):第一步(整段 prompt)照常算完,把这一层的 k、v 存进 `past_kv`(`:123`);之后每步只需算 1 个新 token 的 q/k/v,新的 k、v 用 `torch.cat` 拼到历史后面(`:121-122`,`dim=1` 是序列维)。**只缓存 k/v 不缓存 q**,因为 q 只代表"当前这个 token 要问的问题",算完注意力就没用了;而 k/v 代表"历史 token 可供别人查询的内容",会一直被复用。省下来的正是每步 O(L × 计算) 的重复投影与重复打分。缓存的代价是显存:每层都要存 (batch, 已见长度, 4, 96) 的两张量,长度越长越大。

训练与推理走的是不同分支,这点最值得读:

- 训练(`:111` 的 `past_key_value=None`、`use_cache=False`,例:`trainer/train_pretrain.py:34` 一次喂整段):`seq_len > 1`、无缓存、mask 全 1 → 命中 `:125` 的 flash 分支,`is_causal=True`(`:99`、`:126`)让整段并行算完,一次前向就拿到所有位置的 logits。`use_cache=False` 时 `:123` 直接返回 `past_kv=None`,不占缓存。
- 推理第一步(prefill):同样整段走 flash 分支,但 `use_cache=True` 于是把 k/v 留在手上。
- 推理后续步:`generate` 只喂新 token(`model_minimind.py:270`,`input_ids[:, past_len:]`),`seq_len == 1` 使 `:125` 条件不成立 → 一定走 `:127-131` 的手写分支。此时 `:129` 加的因果遮罩是 1×1 的 `triu(1)`,全是 0,等于不遮——因为"未来"不存在,新 token 可以看见缓存里全部历史。`:130` 的 `attention_mask` 因此必须跟着缓存一起加长(`:271` 每步补一个 1)。
- 位置也要跟着走:`:218` 从缓存长度推出 `start_pos`,`:224` 据此切出这一批 token 该用的 RoPE 角度,不然第 100 个 token 会被当成第 1 个。
- 两者数学上等价:我在本机 `minimind` 环境用默认 768/8 层配置实测,整段一次前向与"1 个 token + 缓存"逐步解码的 logits 最大绝对差约 3e-6(浮点误差量级)。

顺带一个细节:`:216` 会把带 `.layers` 属性的对象(即 transformers 的 `Cache` 类)直接置为 `None`,本模型的缓存格式是"每层一个 (k, v) 元组"的普通 Python 列表(`:234`)。

## 组件速查

| 组件 | 位置 | 说明 |
|---|---|---|
| `RMSNorm` | `model_minimind.py:50-60` | 无中心项的归一:`x * rsqrt(mean(x²)+eps)`(`:57`),对比 LayerNorm 省掉减均值;可学缩放系数初始化为 1(`:54`) |
| RoPE | `precompute_freqs_cis`(`:62-78`)、`apply_rotary_pos_emb`(`:80-84`) | 旋转位置编码;`rotate_half`(`:81`)实现半维配对旋转;`:64-73` 是 YaRN 频率缩放分支,仅当 `end / original_max > 1` 时生效(`:69`) |
| `Attention` | `model_minimind.py:91-134` | GQA:`n_rep = 8//4 = 2`(`:97`),`repeat_kv`(`:86-89`)把 4 份 KV 扩成 8 头;QK-Norm 用 `q_norm`/`k_norm`(`:104-105`,作用在 `:117`);KV Cache 拼接在 `forward`(`:120-123`);flash 走 `F.scaled_dot_product_attention`(`:125-126`),否则手写 mask+softmax(`:128-131`) |
| `FeedForward` | `model_minimind.py:136-146` | SwiGLU:`down(act(gate(x)) * up(x))`(`:146`) |
| `MOEFeedForward` | `model_minimind.py:148-181` | 手写 router(top-k 选专家)+ 专家 FFN + aux load-balancing loss;`aux_loss` 不是 forward 返回值,而是挂在模块属性上(`:176-180`),由 `MiniMindModel.forward` 汇总(`:236`) |
| `MiniMindBlock` | `model_minimind.py:183-199` | pre-norm 残差:`x + attn(norm(x))`(`:192-197`),再 `x + mlp(norm(x))`(`:198`) |
| `MiniMindModel` | `model_minimind.py:201-237` | embedding → N × Block → final norm;持有 RoPE `cos/sin` buffer(`:210-212`)并按 `start_pos` 切片(`:218`、`:224`) |
| `MiniMindForCausalLM` | `model_minimind.py:239-293` | 加 `lm_head`(`:246`),`tie_word_embeddings` 时共享 embedding 权重(`:247`);`forward`(`:250-258`)支持 `logits_to_keep`(`:252-253`,只算最后几个位置的 logits,省显存;`trainer/rollout_engine.py:29` 就是这么用的)和 `labels` 时返回 `loss`+`aux_loss`(`:255-258`);**手写 `generate`**(`:261-293`,自己覆盖了 `GenerationMixin` 的采样逻辑):`temperature`(`:272`)、`repetition_penalty`(`:273-275`)、`top_k`(`:276-277`)、`top_p`(`:278-282`)、`do_sample`(`:283`,关掉就 `argmax` 变贪心)、`eos` 处理与提前退出(`:284-290`)、`streamer`(`:267`、`:287`、`:291`)、`use_cache`(`:270`、`:286`)、`return_kv`(`:292`);`num_return_sequences` 用 `input_ids.repeat(num_return_sequences, 1)` 复制批次实现(`:263`,`attention_mask` 同步 repeat `:264`) |

关于 `num_return_sequences`:机制在本模型 `:263`,但 GRPO 的组采样并不靠它——`trainer/rollout_engine.py:76-81` 是把 prompt 用 `repeat_interleave(num_generations)` 复制后以 `num_return_sequences=1` 调用的,细节见 [10 Rollout 引擎](10_rollout_engine.md)。

## LoRA(`model/model_lora.py`,65 行)

只在这里点名,完整讲解见 [04 LoRA](04_lora.md)。

- `LoRA`(`model_lora.py:6-18`):`B(A(x))` 的低秩旁路(`:18`),`A` 高斯初始化(`:13`)、`B` 零初始化(`:15`)——`B` 全 0 保证刚挂上时旁路输出为 0,模型行为和原权重完全一致。
- `apply_lora()`(`model_lora.py:21-32`):遍历 `named_modules`,给 `in_features == out_features` 的 `nn.Linear`(`:23`)挂 `lora` 并改写 `forward` 为 `原(x) + lora(x)`(`:29-30`),默认 `rank=16`(`:21`)。
- `load_lora`(`model_lora.py:35-42`)/ `save_lora`(`:45-53`):只读写名字里含 `.lora.` 的参数;`merge_lora()`(`:56-65`):把 `B@A` 加进原权重(`:64`)并存一份不含 lora 的 state_dict,用于部署时消除旁路开销。

## 新手常见坑

1. **`apply_lora` 不冻结任何参数**(整个 `model/model_lora.py` 里没有 `requires_grad`),冻结发生在训练脚本里按名字挑参数(`trainer/train_lora.py:131-137`,`'lora' in name` 的放开、其余设 False)。想当然以为 `apply_lora` 之后就只剩 LoRA 在训,是错的。
2. **LoRA 挂不到 k_proj/v_proj 上**:`model_lora.py:23` 的条件是输入维度等于输出维度,而默认 GQA 配置下 `k_proj`/`v_proj` 是 768→384(`model_minimind.py:101-102`)不满足。我用默认配置实测:8 层里只有每层的 `q_proj`、`o_proj` 共 16 个模块被挂上,而不是常见的 q/k/v/o 四路全挂。
3. **`labels` 不要在数据集里预先左移**:模型自己在 `model_minimind.py:256` 用 `logits[..., :-1, :]` 对 `labels[..., 1:]` 做错位。因此数据集给的 `labels` 和 `input_ids` 是**同长且同位**的,只把不算损失的位置填 `-100`(`dataset/lm_dataset.py:56-57`、`generate_labels` `:91-107`),`ignore_index=-100` 在 `:257`。
4. **MoE 的 `aux_loss` 要自己加**:`MOEFeedForward.forward` 只返回 `y`(`model_minimind.py:181`),`aux_loss` 是模块属性(`:176-180`),经 `MiniMindModel` 汇总(`:236`)后放进输出对象(`:258`)。训练脚本必须写成 `res.loss + res.aux_loss`(如 `trainer/train_pretrain.py:35`),否则均衡项等于没生效。

## 阅读建议

从 `MiniMindBlock.forward`(`model_minimind.py:191-199`)入手,30 秒能看懂一层;再往下看 Attention 的 KV cache 分支(`:120-123`)和 `generate` 里"只喂新 token"那行(`:270`)——这是训练/推理共用代码里最值得读的细节。默认配置(768/8 层)建出来的模型 `model.parameters()` 去重后约 63.9M 参数(实测),可以拿它当量纲感觉。

## 下一步

结构看懂了,去 [02 预训练](02_pretrain.md) 看这套骨架是怎么靠"预测下一个 token"从零学起语言知识的。
