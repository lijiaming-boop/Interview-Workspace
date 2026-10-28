# 10-INT8 量化与并行解码教材

> 生成日期：2026-09-13。面向三项任务：跑一遍 INT8 推理并记录、弄懂 INT8 原理、弄懂并行解码原理。
> 所有版本号、脚本路径、基线数字均取自本工作区既有文档与 `01-上游源码` 实测源码，与 `08-后续实验验证` 的验收标准一致。
> Mermaid 图已用 mermaid-cli（mermaid 11）渲染校验通过。
>
> **2026-09-15 修订**：① 新增 [00 速览篇](./00-先读我-量化在VLA中的全景速览.md)，作为 10 分钟入口；
> ② 01 的 §4.4 与 §7 用**实测数据**替换了原先的推测措辞（结论：省显存、慢 2.56 倍）；
> ③ 修正 GL 后端事实（`osmesa` → `egl`，本机无 libOSMesa）。
>
> **第一次来？直接读 [00-先读我·量化在 VLA 中的全景速览](./00-先读我-量化在VLA中的全景速览.md)。**

## 文档清单与阅读顺序

| # | 文档 | 对应任务 | 内容 |
|---|---|---|---|
| **0** | [00-先读我·量化在 VLA 中的全景速览](./00-先读我-量化在VLA中的全景速览.md) | **入口（10 分钟）** | 一页结论 → 量化了谁 → 对称 absmax 的数值直觉 → 三个反直觉事实 → **实测数据** → 六个易错点 → 往下读哪篇 |
| 1 | [01-INT8与FP16量化教程和入门手册](./01-INT8与FP16量化教程和入门手册.md) | 任务 2 + 任务 1 | 数值格式 → 量化数学 → LLM.int8() → 运行时审计 → VLA 敏感度分级 → 动手跑通并记录（含记录模板） |
| 2 | [02-VLA并行解码深入原理与算法](./02-VLA并行解码深入原理与算法.md) | 任务 3 | 自回归瓶颈 → OFT 56 占位 token 并行回归 → FPS/deadline 数学 → 开环执行状态机 → 伪代码 → 与量化交互 |
| 3 | [03-A10部署手册与实验文档框架](./03-A10部署手册与实验文档框架.md) | 任务 1 的执行环境 | 新实例九步装机（版本锁定）→ G0 自检 → INT8 冒烟/500 对照 runbook → 归档框架 |
| 4 | [04-实验运行手册](./04-实验运行手册.md) | 任务 1 的实操 | 可直接粘贴执行：C0 环境自检 → C1 资产核验 → C2/C3 冒烟 → C4 完整 500 episodes → C5 判定，附故障速查 |
| 5 | [05-INT8实验结果报告模板](./05-INT8实验结果报告模板.md) | 任务 1 的交付 | 报告模板（`{{占位符}}` 由 `fill_report.py` 自动填充）+ G0~G5 逐门判定自动生成 |

建议顺序：**0 速览 10 分钟建立全局 → 1 补原理 → 2 理解并行解码 → 3 搭 A10 环境 → 4 逐步执行 → 5 出报告**。
只想尽快开跑的，可以走捷径：**0 → 3 → 4**（跳过 1、2，遇到不懂的再回头查）。

## 实例实测环境（2026-09-14 确认）

教材正文按 cu122 描述，实机 venv 实测为 **torch 2.2.0+cu121**、flash-attn 2.5.8 可正常导入，
A10 capability (8,6)、ABI=False。以实机为准，详见 [04-实验运行手册](./04-实验运行手册.md) 的 C0 自检。

> **GL 后端修正**：本机**没有 `libOSMesa`**，`MUJOCO_GL=osmesa` 会直接报 `glGetError`。
> 必须在 `source activate_oft.sh` **之后**显式 `export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`
> （`_common.sh` 里是 `:-osmesa`，空值绕不过去，必须给非空值）。

## INT8 实测结果（2026-09-15，完整 500 episodes 配对）

```text
峰值显存   15839 MiB（15.47 GiB） → 9315 MiB（9.10 GiB）   省 6524 MiB（−41.2%）  ← 收益确定
平均延迟   217.24 ms → 555.49 ms    speedup 0.391×                              ← 慢了 2.56 倍
deadline 达标率（≤400 ms）  98.4% → 0.0%                                        ← 结构性未达标
动作产出   BF16 36.82 FPS（够 20 Hz，余量 1.84×） / INT8 14.40 FPS（供不上）
成功率     99.2% → 98.8%（损失 +0.4 pp，在 ≤1.0 pp 门槛内）  ← 无观测损失
量化覆盖   436 个 Linear8bitLt（99.8%）｜ language_model 224 / vision_backbone 209 / projector 3
```

一句话：**在这套 VLA 上，INT8 买到的是显存，不是速度。** 数据与判读见
[00 §4](./00-先读我-量化在VLA中的全景速览.md)；完整报告由 `fill_report.py` 生成到 `<RUN_TAG>/INT8实验报告.md`。

## 与既有文档的关系

- 上位验收标准与当前证据边界：[../08-后续实验验证/INT8实验验收标准与文档总结.md](../08-后续实验验证/INT8实验验收标准与文档总结.md)（本教材不重复其 G0~G5 细则，只做讲解与落地）
- 运行脚本说明：[../08-后续实验验证/INT8量化推理说明.md](../08-后续实验验证/INT8量化推理说明.md)
- 历史项目主线：[../00-项目文档时间线.md](../00-项目文档时间线.md)
- FP16 基线：[../04-评测结果/baseline/openvla_oft_fp16_baseline.json](../04-评测结果/baseline/openvla_oft_fp16_baseline.json)

## 关键事实速览（三份文档共用口径）

```text
实验名称（严格）  BF16 baseline vs INT8 主模型 + BF16 action head/proprio projector
平台             魔搭 A10 24G，Python 3.10，venv_oft
基线（FP16 口径） 219.520 ms · policy 4.56 FPS · action 36.44 FPS · 14.963 GiB
并行解码         56 个占位 token 一次 forward → ActionHeadMLP（L1 回归）→ (8,7)
deadline         open_loop=8 @ 20Hz → 400 ms 预算；达标率 ≥95% 才算实时达标
量化入口         run_libero_eval.py --load_in_8bit（bitsandbytes，加载时量化）
动作头红线       永远 BF16/FP16，绝不量化（L1 回归 + 反归一化放大 + 开环累积 + 阈值事件）
成功率门槛       完整 500 episodes 损失 ≤1.0pp 通过；1~2pp 复核；>2pp 失败
冒烟集           4:29, 4:32, 5:35, 6:20, 7:27（只验证能跑通）
量化方式         bitsandbytes LLM.int8()：对称 absmax、zero_point=0；权重逐输出通道 / 激活逐 token
                 离群维度（|x|≥6.0）回退 FP16；checkpoint 不产生新文件（加载时量化）
实测（2026-09-15，完整 500 配对）显存 15.47→9.10 GiB（−41.2%）· 延迟 217.24→555.49 ms（speedup 0.391×）
                                成功 99.2%→98.8%（−0.4pp）· deadline 98.4%→0.0%
量化覆盖（实测）   436 个 Linear8bitLt（99.8%）：language_model 224 / vision_backbone 209 / projector 3
```

## 如何查看 Mermaid 图

- VSCode：装 "Markdown Preview Mermaid Support" 扩展后预览；
- Typora / Obsidian：原生渲染；
- GitHub / Gitea：直接渲染；
- 在线：把代码块内容粘贴到 <https://mermaid.live>。

## 环境速查（详表见文档 3）

```text
torch 2.2.0+cu121 · numpy 1.26.4（锁死） · opencv 4.9.0.80
tensorflow-cpu 2.15.0 · tfds 4.9.3(--no-deps) · tf-metadata 1.15.0 · protobuf 4.25.3
transformers = moojink/transformers-openvla-oft fork(--no-deps) · bitsandbytes 0.43.3
robosuite 1.4.1 · mujoco 3.3.2 · flash-attn 2.5.8(可选，缺则 SDPA)
checkpoint openvla-7b-oft-finetuned-libero-spatial · MUJOCO_GL=egl（不是 osmesa，本机无 libOSMesa）
```
