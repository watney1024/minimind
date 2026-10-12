# 09 知识蒸馏(`trainer/train_distillation.py`,238 行)

## 这一步在干什么(人话)

手里已经有一个表现不错的模型(教师),和一个更小、更容易部署的模型(学生)。蒸馏就是让学生去"抄教师的作业",而不只是抄标准答案。普通微调只告诉学生"这个位置正确的那个词是什么",教师其实还知道"另外几个词也挺合理、只是差一点"——这些信息藏在教师每一步输出的完整概率分布里。把这份分布也拿来当学习目标,学生用同样的数据能多学一层东西。本脚本演示的是 MoE 教师蒸 Dense 学生(`train_distillation.py:138` 注释),输出 `out/full_dist_768.pth`(`:125` + `--save_weight full_dist` `:141`)。

## 先懂这几个词

| 术语 | 大白话 |
|---|---|
| 教师模型 | 提供答案分布的大模型,只被读取、不被训练 |
| 学生模型 | 被训练的那个,通常更小/更简单 |
| logits | 模型对词表里每个词的"原始打分",softmax 之后就是概率 |
| 软标签 | 教师给整个词表的一整排概率,而不只是一个正确词 |
| 硬标签 | 数据集里写死的那个标准答案词(只有 1 个) |
| 温度 T | 把分布"调平"的旋钮:T 越大,次要选项的概率被抬得越高 |
| KL 散度 | 衡量两个概率分布差多少,越小说明学生越像教师 |
| CE(交叉熵) | 只对着硬标签算的常规训练损失 |
| 有效 token | 真正需要模型生成的那部分位置(助手回复),不是提示语 |

## 黑盒 vs 白盒:两种"抄作业"的方式

- **黑盒蒸馏**:只能拿到教师输出的**文本**,把这文本当标准答案做监督微调。它不需要本脚本,本质就是 SFT。README 明确说主线 `full_sft` 数据里"已经混入了相当一部分这样的思路",并点名 DeepSeek R1 / Qwen3 的回答及 tool call、reasoning 数据是黑盒信号(`README.md:737-741`);`sft_t2t` 主线里基于 `qwen3` 合成的采样数据属于此类(`README.md:429`)。
- **白盒蒸馏**:能拿到教师的 **logits**,在 CE 之外再叠一个 KL 项对齐软分布(`README.md:739`)。信息量更大的原因是:硬标签每步只说"就是这个词",软标签给出了教师在所有候选上的相对倾向,学生连带学到"哪些词也不至于错"。

本脚本做的就是白盒。

## 核心实现

### KL loss(`train_distillation.py:25-36`)

标准 Hinton 温度蒸馏。为什么每步都要先除以 T:不除的话教师分布几乎全压在 1 个词上(接近 one-hot),软标签就退化成了硬标签,次要选项的信息等于丢了。

```python
teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()   # :26-27 在 no_grad 里
student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)   # :29
kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')     # :31-35
return (temperature ** 2) * kl                                            # :36 补偿梯度缩放
```

`reduction='batchmean'` 是 KL 的正确平均方式(按 batch 归一,不是按元素)。末尾乘 `T**2` 是因为对 logits 求导时链式法则会带出一个 `1/T²`,不补回来的话一改温度 KL 的实际权重就跟着变小,和 CE 的配比会失衡。

### 混合目标(`train_epoch`,`train_distillation.py:39-133`)

1. **教师先冻住**:`teacher_model.eval()` + `requires_grad_(False)`(`:42-44`,主流程 `:199-200`)。不这么做,教师会在反向传播里被当参数更新,而它本该是固定的参照物,而且白显存也白算梯度。
2. **教师前向**:同一批 `input_ids` 喂给教师,取 `logits[..., :-1, :]`(`:62`)。因为教师/学生都固定用同一套 MiniMind tokenizer(`trainer/trainer_utils.py:122`),词表天然对齐;`:63-64` 只是把教师 logits 的最后一维截到学生词表大小做保护。
3. **只在有效 token 上算**:数据里提示语部分的 label 是 `-100`,`loss_mask = (labels[..., 1:] != -100).float()`(`:49`)。教师对提示语那几个字的"偏好"没意义——那部分是给定内容,不该学生去学。CE 和 KL 都先用 `loss_mask_flat == 1` 展平筛选再计算(`:76`、`:82-86`)。
4. **CE loss**:`F.cross_entropy`(`:70-78`),label 右移一位与 logits 对齐(`:68`),`reduction='none'` 是为了能逐位置乘 mask。
5. **总 loss**(`:91`):**`alpha*CE + (1-alpha)*KL`**。默认 `--alpha 0.5`(`:163`)、`--temperature 1.5`(`:164`,help 里推荐 1.0–2.0)。保留 CE 是防止学生只拟合教师、把真实答案也带偏(教师错的地方学生也照抄)。
6. **MoE 学生额外加 `aux_loss`**(`:77`):路由负载不均衡的惩罚,与蒸馏无关,是 MoE 本身要交的账。
7. 保存:每 `--save_interval` 步落盘 fp16 state_dict(`:122-130`)。

## 教师/学生配置(`train_distillation.py:140-167`)

| 参数 | 默认 | 行号 | 说明 |
|---|---|---|---|
| `--teacher_hidden_size` / `--teacher_num_layers` | 768 / 8 | :156-157 | 教师结构 |
| `--student_hidden_size` / `--student_num_layers` | 768 / 8 | :154-155 | 学生结构 |
| `--teacher_use_moe` / `--student_use_moe` | 1 / 0 | :159-158 | 默认 MoE 教师 + Dense 学生 |
| `--from_student_weight` | `full_sft` | :160 | **学生起点权重**(注意参数名) |
| `--from_teacher_weight` | `full_sft` | :161 | 教师权重,解析为 `out/full_sft_768moe.pth` |
| `--alpha` / `--temperature` | 0.5 / 1.5 | :163-164 | 混合系数与温度 |
| `--learning_rate` | 5e-6 | :144 | 蒸馏是小 lr 精调 |
| `--epochs` / `--batch_size` | 6 / 32 | :142-143 | |
| `--data_path` | `../dataset/sft_t2t_mini.jsonl` | :153 | 与 SFT 用同一批对话数据 |
| `--save_weight` | `full_dist` | :141 | 产出文件名前缀 |

权重的实际文件名由配置拼出来:`{save_dir}/{from_weight}_{hidden_size}{_moe}.pth`(`trainer/trainer_utils.py:127`),所以教师写 `full_sft` 而学生写 `full_sft` 并不会撞车:教师带 `_moe` 后缀,学生不带。注意**没有** `--use_moe` 这个参数,只有上面带前缀的两个。

## 运行

```bash
cd trainer && python train_distillation.py                 # 单卡
torchrun --nproc_per_node 1 trainer/train_distillation.py  # README 约定写法,README.md:755-757
```

想演示"大蒸小"就把 `--teacher_hidden_size` 调大,但此时教师 ckpt 文件名也跟着变,得确有其文件。

## 新手常见坑

1. **权重名记错**:学生起点是 `--from_student_weight`,不是 `--from_weight`(`:160`、`:196`)。
2. **加载是 `strict=False`**(`trainer/trainer_utils.py:129`):教师结构写错(层数/hidden 与 ckpt 不符)不报错,部分参数静默保持随机初始化,于是 KL 在往一个乱说话的教师靠,现象是 loss 不降。务必核对日志打印的教师参数量(`:201`)。
3. **默认教师和学生几乎同尺寸**(都 768/8):收益来自 MoE vs Dense 的容量差,不是参数量差,别以为默认就是"大模型蒸小模型"。
4. **alpha 与 temperature 联动**:T 改变分布平坦度(`:36` 只补偿了梯度尺度),而 `--alpha 1` 会把 KL 项直接清零(`:91`),退化成普通 SFT。

## 下一步

蒸馏产出的 `full_dist_768.pth` 换到 `--from_weight` 就能拿去评测或上线,见 `11_eval_deploy.md`;想让模型自己生成数据(黑盒蒸馏的教师侧)见 `10_rollout_engine.md`。
