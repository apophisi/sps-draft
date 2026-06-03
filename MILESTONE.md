# 阶段性进展报告：Speculative Decoding 原型

日期：2026-05-27

## 1. 项目目标

本项目旨在复现 speculative decoding 的基础推理框架，并在此基础上实现
dynamic draft depth，即动态调整每轮 draft token 数量的策略。

当前模型配置：

- Draft model：`Qwen/Qwen3-0.6B`
- Target model：`Qwen/Qwen3-1.7B`
- 技术栈：PyTorch、Transformers、NumPy、uv

当前阶段的重点是先保证算法流程正确、指标可观测、实验结果可复现。系统级性能优化仍是后续工作重点。

## 2. 当前进度

### 2.1 基础运行框架

已完成：

- draft model 和 target model 的加载
- tokenizer 加载与 prompt 编码
- prompt prefill，并保存 KV cache
- 基于 KV cache 的单 token decode
- 基于 KV cache 的多 token decode：`ModelRunner.decode_many`
- Hugging Face 下载源配置
  - 默认镜像：`https://hf-mirror.com`
  - 可通过 `--hf-endpoint` 临时切换下载源
  - 可通过 `--local-files-only` 使用本地缓存模型

相关文件：

- `runtime/model.py`
- `runtime/tokenization.py`
- `runtime/deps.py`
- `pipeline/prefill.py`

### 2.2 固定 K 的 Speculative Decoding

已实现完整的 fixed-K speculative decoding 流程：

1. draft model 先提出 `K` 个 token。
2. target model 对这 `K` 个 token 进行验证。
3. 对第 `i` 个 draft token `x_i`，接受概率为：

   ```text
   min(1, p_i(x_i) / q_i(x_i))
   ```

4. 如果某个 token 被拒绝，则从残差分布中采样一个校正 token：

   ```text
   normalize(max(p_i - q_i, 0))
   ```

5. 如果 `K` 个 token 全部接受，则从 target model 额外采样一个 bonus token。
6. 生成过程在达到 `max_new_tokens` 或遇到 EOS 时停止。

当前 target 验证已经改为 batched verification：每轮 draft 完成后，target 对整段 draft token 做一次 forward，批量得到所有验证位置的 logits。

主要输出指标：

- `avg_accept`：每轮平均接受的 draft token 数
- `accept rate`：总接受 draft token 数 / 总 proposed draft token 数
- round 数
- 生成 token 数
- 是否因为 EOS 停止

相关文件：

- `speculative/proposal.py`
- `speculative/verification.py`
- `speculative/generation.py`
- `speculative/sampling.py`
- `runtime/model.py`
- `main.py`

运行示例：

```bash
uv run python main.py -k 4 --max-new-tokens 128 --local-files-only
```

### 2.3 Dynamic K 策略

已实现第二阶段实验所需的 dynamic draft depth 策略。

固定 K baseline：

- `K=4`
- `K=8`

动态 K 策略：

- `p_max` early stopping
  - 当 draft model 当前最大概率低于阈值时停止继续 draft
  - 当前测试阈值：`0.6`、`0.7`、`0.8`
  - `K_max = 8`
- 可选的 top1-top2 margin 策略
  - 当 `p_top1 - p_top2` 低于给定 margin 时停止继续 draft

相关文件：

- `dynamic_policy.py`
- `run_dynamic_k.py`
- `prompts/default_prompts.txt`
- `results/dynamic_k_results.jsonl`

实验命令：

```bash
uv run python run_dynamic_k.py --overwrite --local-files-only
```

查看结果汇总：

```bash
uv run python run_dynamic_k.py --summary-only
```

## 3. 初步实验结果

当前结果来自 `results/dynamic_k_results.jsonl`。

当前 prompt 集合上的平均结果如下：

| 方法 | 策略 | Tokens/s | Speedup | Acceptance Rate | Avg Accept Length | Avg Draft Length |
|---|---:|---:|---:|---:|---:|---:|
| Fixed K | K=4 | 6.55 | 0.200 | 0.134 | 0.531 | 3.96 |
| Fixed K | K=8 | 3.82 | 0.116 | 0.052 | 0.414 | 7.91 |
| Dynamic K | p_max > 0.6 | 9.74 | 0.302 | 0.355 | 0.577 | 1.55 |
| Dynamic K | p_max > 0.7 | 10.18 | 0.313 | 0.445 | 0.538 | 1.20 |
| Dynamic K | p_max > 0.8 | 11.30 | 0.348 | 0.563 | 0.759 | 1.30 |

这里的 `speedup` 定义为：

```text
speculative tokens/s / AR baseline tokens/s
```

因此，只有当 `speedup > 1.0` 时，才表示 speculative decoding 真正快于 target-only autoregressive baseline。当前所有结果仍低于 `1.0`，说明当前原型还没有实现端到端加速。

不过，从 fixed K 到 dynamic K 的相对提升已经比较明显：

- batched target verification 相比早期逐 token target verification 明显提升了吞吐。
- Dynamic K 的 tokens/s 高于 fixed K。
- Fixed `K=8` 比 fixed `K=4` 更慢，主要原因是 proposed token 较多，但接受率较低，浪费了更多 draft 工作。
- Dynamic K 将平均 draft length 控制在约 `1.2-1.6`，减少了无效 proposal。
- 当前最好结果是 `p_max > 0.8`：
  - `tokens/s = 11.30`
  - `acceptance rate = 0.563`
  - `speedup = 0.348`
- 相比 fixed `K=4` 的 `speedup = 0.200`，dynamic K 的相对效率有明显改善。

## 4. AR Baseline 说明

当前 AR baseline 是 target-only autoregressive decoding：

- 只使用 `Qwen/Qwen3-1.7B`
- 使用同一个 prompt prefill 后的 KV cache
- 不计入 prefill 时间，只统计 decode 阶段吞吐
- 与 speculative decoding 使用相同的 target sampling 设置
- 记录字段包括：
  - `ar_tokens`
  - `ar_elapsed_sec`
  - `ar_tokens_s`

该 baseline 是当前 Python 原型内的合理对照对象，但不是 vLLM、TensorRT-LLM 等高度优化推理引擎下的系统级 baseline。

## 5. 当前限制

- 当前实现仍以清晰和可解释为主，并非高度优化的推理系统。
- 虽然 target 验证已经 batched，但 draft proposal 仍是逐 token 生成。
- Python 层循环、cache 同步和实验脚本开销仍然较大。
- 当前 prompt 集合较小，结果对 prompt 分布较敏感。
- Qwen3-0.6B 与 Qwen3-1.7B 的分布差异仍可能较大，导致 acceptance length 不够高。
- 当前 speedup 仍小于 1，尚未超过 AR baseline。
- dynamic K 策略仍是启发式规则，还没有经过充分调参。

## 6. 下一步计划

### 6.1 正确性测试

计划增加单元测试，覆盖：

- acceptance probability
- residual sampling
- EOS stopping
- `max_new_tokens` 边界行为
- dynamic K stopping rule
- batched verification 与 sequential verification 的一致性

此外，可以构造 toy model，对比 speculative decoding 与 target-only sampling 的分布一致性。

### 6.2 性能优化

后续优化方向：

- 继续减少 Python-side overhead
- 分离并记录更细粒度的时间：
  - draft proposal time
  - target verification time
  - cache update time
  - total decode time
- 优化 draft cache 推进方式
- 减少不必要的 CPU/GPU 同步
- 尝试更适合 speculative decoding 的 draft/target 组合

### 6.3 实验扩展

后续实验计划：

- 扩大 prompt 集合规模
- 按任务类型拆分 prompt：
  - 短回答
  - 长解释
  - 创作
  - 代码
  - 推理/数学
  - 中英混合
- 测试更多 `p_max` threshold
- 跑 top1-top2 margin 策略并与 `p_max` 策略对比
- 后续可接入 SPEED-Bench 或 Spec-Bench 的 prompt 集合作为更标准的 benchmark 输入

### 6.4 结果展示

计划增加图表和导出：

- 不同策略的 tokens/s
- 不同策略的 acceptance rate
- 不同策略的 average draft length
- 每个 prompt 的 speedup 分布
- AR baseline 与 SPS 方法的对比表

原始实验日志继续保存在：

```text
results/dynamic_k_results.jsonl
```

## 7. 阶段状态

已完成：

- 基础模型加载与 prefill 框架
- fixed-K speculative decoding
- accept/reject correction
- batched target verification
- EOS 与 `max_new_tokens` 停止
- `avg_accept`、`accept rate` 等运行指标
- AR baseline 直观字段记录
- dynamic K 策略
- dynamic K 实验脚本
- 初步实验结果与 summary 输出

进行中：

- benchmark 方法完善
- dynamic policy 效果分析
- speedup 未达 1 的瓶颈定位

计划中：

- 更完整的单元测试
- 更大规模 prompt benchmark
- 更细粒度性能 profiling
- 实验结果图表化
