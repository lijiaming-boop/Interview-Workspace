chunk 8 - control

# Action-chunk diagnostic summary

- Episodes: 3
- Successes: 2
- Failures: 1

| Task | State | Hz | Open-loop | Success | First close | First transfer | Deadline met | Forced replans |
|---:|---:|---:|---:|---|---|---|---:|---:|
| 4 | 32 | 20 | 8 | True | c5/a1 | c5/a1 | 93.3% | 0 |
| 6 | 20 | 20 | 8 | True | c6/a1 | c6/a1 | 100.0% | 0 |
| 7 | 27 | 20 | 8 | False | c4/a6 | c4/a6 | 100.0% | 0 |

> `First close` and `First transfer` are action-log heuristics, not ground-truth grasp labels.
> A 50 Hz claim is supported only when `deadline_met_rate` is near 100% on the target GPU.



chunk 4

# Action-chunk diagnostic summary

- Episodes: 3
- Successes: 1
- Failures: 2

| Task | State | Hz | Open-loop | Success | First close | First transfer | Deadline met | Forced replans |
|---:|---:|---:|---:|---|---|---|---:|---:|
| 4 | 32 | 20 | 4 | False | c10/a3 | c10/a3 | 0.0% | 0 |
| 6 | 20 | 20 | 4 | True | c12/a1 | c12/a1 | 0.0% | 0 |
| 7 | 27 | 20 | 4 | False | c10/a1 | c10/a1 | 0.0% | 0 |

> `First close` and `First transfer` are action-log heuristics, not ground-truth grasp labels.
> A 50 Hz claim is supported only when `deadline_met_rate` is near 100% on the target GPU.



camera -w1

# Action-chunk diagnostic summary

- Episodes: 2
- Successes: 1
- Failures: 1

| Task | State | Hz | Open-loop | Success | First close | First transfer | Deadline met | Forced replans |
|---:|---:|---:|---:|---|---|---|---:|---:|
| 4 | 29 | 20 | 8 | False | c5/a1 | c5/a1 | 0.0% | 0 |
| 5 | 35 | 20 | 8 | True | c3/a5 | c3/a5 | 6.7% | 0 |

> `First close` and `First transfer` are action-log heuristics, not ground-truth grasp labels.
> A 50 Hz claim is supported only when `deadline_met_rate` is near 100% on the target GPU.



# Action-chunk diagnostic summary

- Episodes: 2
- Successes: 1
- Failures: 1

| Task | State | Hz | Open-loop | Success | First close | First transfer | Deadline met | Forced replans |
|---:|---:|---:|---:|---|---|---|---:|---:|
| 4 | 29 | 20 | 8 | False | c5/a2 | c5/a2 | 0.0% | 0 |
| 5 | 35 | 20 | 8 | True | c3/a5 | c3/a5 | 0.0% | 0 |

> `First close` and `First transfer` are action-log heuristics, not ground-truth grasp labels.
> A 50 Hz claim is supported only when `deadline_met_rate` is near 100% on the target GPU.

腕部相机输入和预处理正常。将腕部相机权重由1.0提高至1.25后，Task 4 / State 29仍失败，Task 5 / State 35仍成功，成功率保持1/2。Episode 230失败并非主要由相机权重不足导致，而是策略在未稳定抓住目标的情况下继续执行搬运，下一阶段应实现基于仿真接触和目标位移的抓取确认与失败后强制重规划。



# Action-chunk diagnostic summary

- Episodes: 2
- Successes: 1
- Failures: 1

| Task | State | Hz | Open-loop | Success | First close | First transfer | Deadline met | Grasp checks | Empty-grasp replans | Forced replans |
|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| 4 | 29 | 20 | 8 | False | c5/a1 | c5/a1 | 96.4% | 0 | 0 | 0 |
| 5 | 35 | 20 | 8 | True | c3/a5 | c3/a5 | 100.0% | 0 | 0 | 0 |

> `First close` and `First transfer` are action-log heuristics, not ground-truth grasp labels.
> A 50 Hz claim is supported only when `deadline_met_rate` is near 100% on the target GPU.



# Action-chunk diagnostic summary

- Episodes: 2
- Successes: 1
- Failures: 1

| Task | State | Hz | Open-loop | Success | First close | First transfer | Deadline met | Grasp checks | Empty-grasp replans | Forced replans |
|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| 4 | 29 | 20 | 8 | False | c5/a1 | c5/a1 | 97.5% | 19 | 19 | 19 |
| 5 | 35 | 20 | 8 | True | c3/a5 | c3/a5 | 100.0% | 2 | 1 | 1 |

> `First close` and `First transfer` are action-log heuristics, not ground-truth grasp labels.
> A 50 Hz claim is supported only when `deadline_met_rate` is near 100% on the target GPU.

![image-20260831033954278](C:\Users\27013\AppData\Roaming\Typora\typora-user-images\image-20260831033954278.png)

Episode 230 的目标选择正确，抓取检测并非阈值误判。全部19次检测均无双指接触，目标在第一次轻微扰动后保持静止，而夹爪与目标距离持续从0.129米扩大到0.389米，证明策略在空抓后继续远离目标。当前仅清空动作队列无法恢复，下一步需要在首次闭合时阻断transfer，并加入闭合确认、强制张开、重新观察及最多3次重试的恢复状态机。