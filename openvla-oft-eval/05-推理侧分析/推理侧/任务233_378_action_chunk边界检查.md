# 任务 233 / 378：action-chunk 边界检查

## 可复现实例

| 全局 episode | LIBERO task_id | 初始状态下标 | 场景 | 结果 |
|---:|---:|---:|---|---|
| 233 | 4 | 32 | top drawer | 失败 |
| 378 | 7 | 27 | stove | 失败 |

两条记录均使用 `NUM_ACTIONS_CHUNK=8`、20 Hz、`open_loop=8`。单个完整 chunk 的开环时间是 400 ms。

## 已有实验日志能够确认的事实

| episode | 首次闭合命令 | 旧日志中的“首次 transfer” | 成功 |
|---:|---|---|---|
| 233 | chunk 5 / action 1 | chunk 5 / action 1 | 否 |
| 378 | chunk 4 / action 6 | chunk 4 / action 6 | 否 |

因此，两次首次闭合都发生在 chunk **内部**，不是边界：233 在该 chunk 剩余 6 个动作，378 剩余 1 个动作。

旧版 `First transfer` 规则把“夹爪闭合且包含平移”的动作也算作 transfer，所以它与 close 相等是规则导致的，不能据此证明真正的 grasp→transfer 切换发生在 chunk 边界或同一动作。视频只能补充说明两例均为空抓后仍执行了搬运/放置；原始 `chunks.jsonl` 未保留在当前工作区，无法反推出每个 8×7 action。

## 结论：能否只在边界插入检测

不能依赖“仅在固定 chunk 边界检查”。要使边界检查在真正 transfer 前生效，必须同时满足：

1. 首次闭合位于较早的 chunk；
2. 首个闭合后的搬运动作是后续 chunk 的 `action 0`。

若搬运动作还在闭合所在 chunk 的剩余位置，边界检查到来前已执行了开环尾部动作。由于 233 的闭合在 `a1`，风险窗口尤其大；378 在 `a6`，仍有一次开环尾部动作。两例应在**首次 close 动作被取出时**立即清空队列、保持闭合两步、使用仿真接触和 gripper-target 距离验证，再决定是否重新规划。该路径已在仿真 runner 的 `--verify_grasp True` 中实现；仅用于仿真真值，不可直接部署到真机。

## 已完善的诊断输出

`analyze_action_chunks.py` 现在区分：

- `first_close_*`：首次执行的闭合命令；
- `first_close_with_motion_*`：闭合命令是否把最后接近与平移绑定；
- `first_transfer_*`：闭合之后、另一条执行动作上的首次闭合搬运；
- `boundary_check_before_transfer`：仅当该搬运是后续 chunk 的 `action 0` 时为 `True`。

这避免了旧版“close 本身被误标为 transfer”的歧义，并可从每次回放保存的 `chunks.jsonl`（完整 8×7 原始/环境 action、gripper 命令、边界接触与目标位姿）和 `steps.jsonl` 得到结论。

## 仿真复现实验

应先在含 OpenVLA-OFT 权重的 `oft` 环境执行：

```bash
RUN_TAG=boundary-233-378 bash experiments/robot/libero/failure_study/run_simulation_study.sh boundary8
RUN_TAG=boundary-233-378-verify bash experiments/robot/libero/failure_study/run_simulation_study.sh grasp_recovery_boundary
```

两条命令都会重放 233/378（`4:32,7:27`）；第一条保存完整 chunk，第二条在首次闭合时启动真值抓取确认。验收条件是：`boundary_check_before_transfer` 的真假、`grasp_close_intercepted`、`grasp_verified` 或 `empty_grasp_forced_replan` 事件，以及视频中空夹爪是否还进入盘子上方。

本机未能实际启动回放：仅有 base/graph 环境，二者均缺少 `torch`；也未发现 `openvla-7b-oft-finetuned-libero-spatial` checkpoint。因此本报告不把已有汇总表外推为新的仿真结果。
