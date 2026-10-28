# Bridge → Isaac Lab 迁移掉点分析报告

## 1. 实验对象

任务固定为：Franka 从官方 Isaac Lab cube-lift 桌面场景抓起单个 cube，放到绿色目标区域。每个 episode 使用相同的初始状态 seed，控制周期为 20 Hz，默认 `open_loop_steps=1`，即每步重新渲染 RGB 并重新推理。

模型输入是现有项目的 Bridge action head/checkpoint：

```text
RGB → OpenVLA-OFT → [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
    → [IK pose 6D, BinaryJointPosition gripper 1D]
    → env.step()
```

## 2. 成对评测协议

必须使用相同：checkpoint、`bridge_orig`/`unnorm_key`、episode seed、目标物初始状态、`num_envs`、`open_loop_steps` 和最大步数。只改变视觉域处理：

| 条件 | 相机 | RGB | stage 随机化 | 目的 |
|---|---|---|---|---|
| raw | Isaac 固定相机 | 原始 RGB | 无 | 迁移前基线 |
| camera_aligned | Bridge 风格固定位姿 | 4:3 中心裁剪、曝光对齐 | 无 | 分离视角/构图因素 |
| bridge_aligned | 同上 | 亮度、对比度、饱和度、轻微噪声 | 灯光/材质/相机小范围随机 | 补救后的鲁棒性 |

运行：

```bash
bash 12-IsaacLab-Bridge-Franka/run_domain_gap_eval.sh \
  /path/to/openvla-bridge-checkpoint 50 8
```

脚本分别写入 `episodes.jsonl`、`summary.json` 和闭环 demo MP4，再生成 `domain_gap_report.md`。当前仓库未安装 Isaac Lab、Isaac Sim 和 PyTorch，因而本报告不填入伪造的成功率；真实结果必须由上述命令产生。

## 3. 结果表

运行后由 `analyze_results.py` 自动填充：

| 条件 | n | 成功 | 成功率 | 95% CI | ever-lifted |
|---|---:|---:|---:|---:|---:|
| raw | 待实跑 | 待实跑 | 待实跑 | 待实跑 | 待实跑 |
| camera_aligned | 待实跑 | 待实跑 | 待实跑 | 待实跑 | 待实跑 |
| bridge_aligned | 待实跑 | 待实跑 | 待实跑 | 待实跑 | 待实跑 |

掉点定义：

```text
迁移前掉点 = raw_success_rate - Bridge/真实数据参考成功率
视觉补救收益 = bridge_aligned_success_rate - raw_success_rate
```

若没有同 checkpoint 在 Bridge/真实数据上的参考成功率，则只报告 Isaac Lab 内部 raw → aligned 的差异，不把它命名为 sim-to-real 绝对掉点。

## 4. 失败归因规则

- `ever_lifted=False`：抓取失败。优先看 RGB 中 cube 的像素位置、夹爪接近高度、动作前 6 维单位/限幅和 `+1=open/-1=close` 是否反了。
- `ever_lifted=True` 且未放置：搬运或放置失败。优先看 `bridge_frame`/`ik_frame`、目标区坐标、释放时机和 action chunk 是否过长。
- raw 失败而 aligned 成功：域差距证据，继续比较相机位姿、光照、材质和 RGB 统计。
- raw 与 aligned 都失败：先用 `--policy heuristic` 验证 IK/夹爪/场景，再排查 checkpoint 的 `unnorm_key` 和 `dataset_statistics.json`。
- heuristic 也失败：不应继续调 VLA；先修 controller、camera path、Franka joint/body name 或 Isaac Sim 物理参数。

## 5. 动作空间验收清单

- [x] Bridge action 为 7D，前 6 维是 EEF delta pose，末维是 gripper。
- [x] Isaac Lab arm action 为 6D relative pose，使用官方 DLS differential IK。
- [x] Bridge 的绝对夹爪 `0=close, 1=open` 映射为 Isaac Lab binary joint 的 `-1/+1`。
- [x] 位移/旋转限幅在动作下发前执行，避免异常反归一化造成穿模。
- [x] body/root 坐标变换有单测，默认 Panda base 对齐时不做额外变换。
- [ ] Isaac Lab headless smoke 通过。
- [ ] OpenVLA raw / aligned 成对评测完成。
- [ ] demo MP4、episodes.jsonl 和最终掉点百分比归档。

## 6. 结论模板

实跑完成后只保留下面这种证据链：

```text
raw:             __ / __ = __.__%
camera_aligned:  __ / __ = __.__%  (vs raw: __.__ pp)
bridge_aligned:  __ / __ = __.__%  (vs raw: __.__ pp)

主要掉点阶段：抓取 / 搬运 / 放置 / 推理超时 / 物理碰撞
最有效补救：相机位姿 / 灯光材质 / RGB 随机化 / 动作限幅 / 坐标系 / chunk 长度
```
