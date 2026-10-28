# 本项目 Agent 开发知识整理

> 范围：中央空调 / 空压机节能测算项目（`conditioner_saving`）中所有与 Agent 开发相关的
> 设计、代码、工程实践与经验教训。
>
> 整理日期：2026-09-09 ｜ 代码基线：`agent/` 包 v2（自研 ReAct 编排器）
>
> 配套文档：[ARCHITECTURE.md（决策边界）](../ARCHITECTURE.md) ·
> [从Skill到Agent升级改造设计.md（演进全过程）](../从Skill到Agent升级改造设计.md) ·
> [evaluation/report.md（评测报告）](../evaluation/report.md)

---

## 目录

1. [一句话定位与演进脉络](#一-一句话定位与演进脉络)
2. [架构分层与代码地图](#二-架构分层与代码地图)
3. [机制一：ReAct 编排循环](#三-机制一react-编排循环)
4. [机制二：白名单硬约束（本项目的核心创新点）](#四-机制二白名单硬约束本项目的核心创新点)
5. [机制三：Brain 双实现（LLM / 规则）](#五-机制三brain-双实现llm--规则)
6. [机制四：工具注册表与确定性工具](#六-机制四工具注册表与确定性工具)
7. [机制五：控制类工具与 Human-in-the-loop](#七-机制五控制类工具与-human-in-the-loop)
8. [机制六：轨迹可观测性](#八-机制六轨迹可观测性)
9. [机制七：持久化、断点恢复与并发锁](#九-机制七持久化断点恢复与并发锁)
10. [Skill 体系：SKILL.md 与 claude-agent-sdk](#十-skill-体系skillmd-与-claude-agent-sdk)
11. [LLM 网关：成本治理五件套](#十一-llm-网关成本治理五件套)
12. [评测体系：把"正确率"变成可回归的门禁](#十二-评测体系把正确率变成可回归的门禁)
13. [平台化工程能力清单](#十三-平台化工程能力清单)
14. [提炼：可复用的 Agent 开发方法论（15 条）](#十四-提炼可复用的-agent-开发方法论15-条)
15. [踩过的坑与修复清单](#十五-踩过的坑与修复清单)
16. [命令速查与代码索引](#十六-命令速查与代码索引)

---

## 一、一句话定位与演进脉络

**定位**：一个"LLM 只当大脑、手脚全是确定性函数"的业务 Agent——
**算得准的交给公式，看得懂的交给模型**；凡是进入报价的数字，必须来自确定性引擎。

### v1 → v2 的两次转身

| 阶段 | 形态 | 关键文件 |
|---|---|---|
| v1 早期 | Streamlit + `subprocess` 调 Claude CLI + 3 个 Skill | `core/skill_caller.py` |
| v1 中期 | 迁移到 `claude_agent_sdk.query()`，每个业务动作起一个**完整 Agent 会话**（带 Bash/Read/Write），流程顺序写死在五阶段状态机里 | 同上 |
| v1 终态（2026-07） | Next.js + FastAPI 重构，全部 LLM 调用收拢到 SDK，Skill 演进到 12 个 | commit `21ef5d4` |
| **v2（2026-08-29，`c307ade`）** | **自研 `agent/` 包：进程内 ReAct 编排器 + 白名单硬约束 + Brain 双实现 + 结构化轨迹 + 评测门禁** | `agent/` |

### v1 的六个结构性问题（升级动机，值得反复对照自查）

1. **为"执行一条 python 命令"付整个 agent 会话的钱**——纯规则脚本也要起会话、加载 SKILL.md、展开工具集。
2. **编排写死不可演进**——流程顺序在 UI/服务层 if-else 里，换流程 = 改代码，模型无法根据观察动态改变路径。
3. **安全面失控**——会话内模型持有 Bash/Write，处理用户上传文件 ≈ 把任意代码执行暴露给不可信输入。
4. **不可观测**——只有 ai_logs 文本流，答不出"模型当时为什么这么做、每步多少 token"。
5. **不可测试**——起会话强依赖 API Key，无 Key 直接报错，单测全靠 mock。
6. **无持久化、无恢复**——会话是进程内 dict；失败静默降级 mock 假数据（**返回假节能率**，最危险的一条）。

### v2 的收益（有数据支撑）

| 指标 | v1 | v2 |
|---|---|---|
| 一次完整测算 LLM 成本 | 3 个会话、数十次内部模型调用 | **5 次决策调用 ≈ 3.5k tokens**（in 2,966 / out 504）；评测/离线模式 **0 token** |
| 流程决策正确率 | 无法度量 | **100%（9/9）**，评测回归门禁 |
| 硬约束 | 依赖模型自觉 | 白名单在代码层强制，模型绕不过 |
| 可观测 | 文本流 | 结构化 `agent_trace.json` + 落库 + SSE 逐帧回放 |
| 可测试 | 必须 mock | RuleBrain 零成本复现 |

---

## 二、架构分层与代码地图

```
┌──────────────────────────────────────────────────────────────┐
│ 接入层   FastAPI /agent/run (SSE) │ 分阶段兼容端点 │ Next.js   │
├──────────────────────────────────────────────────────────────┤
│ 编排层   AgentOrchestrator：ReAct 循环 + 白名单硬约束          │
│          Brain: LLMBrain（JSON 强约束）│ RuleBrain（零成本）   │
├──────────────────────────────────────────────────────────────┤
│ 工具层   ToolRegistry：extract/suggest/apply/calculate + 控制 │
├──────────────────────────────────────────────────────────────┤
│ 引擎层   formulas Excel 引擎 │ 规则提取/质检 │ 气象服务        │
├──────────────────────────────────────────────────────────────┤
│ 基础设施  SessionManager │ trace 落盘 │ LLM 网关 │ 事件总线    │
└──────────────────────────────────────────────────────────────┘
```

| 组件 | 文件 | 职责 |
|---|---|---|
| ReAct 循环 + 白名单 | `agent/orchestrator.py` | 主循环、门禁计算、状态视图、事件回调 |
| LLM 决策 + JSON 重试 | `agent/brain.py::LLMBrain` | 单次 JSON 决策调用 |
| 零成本规则策略 | `agent/brain.py::RuleBrain` | 评测/离线/无 Key 场景 |
| 工具注册表 + handler | `agent/tools.py` | JSON Schema 工具定义、确定性执行 |
| 轨迹记录 | `agent/trace.py` | 逐步落盘 `agent_trace.json` |
| Skill 调用器（SDK 桥） | `core/skill_caller.py` | LLM 型技能、确定性脚本直调、沙箱 |
| LLM 客户端 / 网关 | `core/llm_client.py` / `core/llm_gateway.py` | 计量、预算、熔断、重试、路由 |
| 服务层桥接 | `apps/api/agent_service.py` | 选 Brain、起线程、落库、恢复、HITL |
| 持久化 | `apps/api/storage.py` / `models.py` | 8 张表、写透 LiveSession |
| 评测 | `evaluation/` | 数据集、指标、基线比对 |
| 前端 | `apps/web/src/app/agent/page.tsx`、`AgentRunPanel.tsx`、`AgentTraceView.tsx` | 轨迹逐帧回放 |

---

## 三、机制一：ReAct 编排循环

`AgentOrchestrator.run()` 每一步的固定顺序（`agent/orchestrator.py:105`）：

```
① abort_check()      锁租约检查——锁易主则抛 LockLostError 协作式退出
② _allowed_tools()   计算白名单（硬约束）+ 门禁说明 gate_note
③ _state_view()      生成给模型的紧凑状态摘要
④ brain.decide()     决策：返回 BrainDecision(tool, args, reason, llm_meta)
⑤ 越权校验           白名单外的工具 → 拒绝，错误作为 observation 反馈
⑥ 执行 or 拦截       确定性工具 → registry.execute()；控制类工具 → 改状态并 break
⑦ trace + on_step    落盘轨迹 + 落库
⑧ 熔断判断           连续 3 次工具异常 → 中止；超过 max_steps → 强制结束
```

**几个容易忽略但很关键的实现细节：**

- **被拒绝的步骤也要落库**（`orchestrator.py:155`）。模型越权那一步同样消耗了决策 token，
  漏掉会导致数据库 token 口径少计一次。
- **观察结果（observation）是压缩过的**——只回传关键字段（完整率、置信度、节能率区间、
  前 5 条 critical issues），不把完整建议载荷塞回 prompt，这是省 token 的关键。
- **`extract_survey` 入参由编排器强制覆盖**（`orchestrator.py:161`）——路径以会话记录为准，
  不让模型自己编造文件路径。
- **状态完全由产物驱动**：`_allowed_tools()` 只看 `ctx` 里的产物，不依赖外部变量，
  这是断点恢复能成立的前提。

---

## 四、机制二：白名单硬约束（本项目的核心创新点）

**设计信条：软决策给自主性，硬约束给可靠性。**
模型可以在白名单内自由选择，但白名单本身由代码计算——**模型绕不过**。

`_allowed_tools()` 返回 `(白名单, 门禁说明)`，门禁说明会作为 `gate_note` 进状态视图，
告诉模型"为什么现在只能做这个"。

| 约束 | 触发条件 | 强制行为 |
|---|---|---|
| 工具前置依赖 | 未提取 / 未建议 / 未填充 | 白名单中不出现下游工具 |
| 质量门禁 | `confidence_level == "low"`（完整率 <50%） | 只剩 `request_user_input`，流程以 `needs_input` 终止 |
| 站房类型拦截 | `station_type == "unsupported"`（热水泵/蓄冷/中低温机组） | 同上，附识别依据 |
| 不可填充拦截 | 规则引擎 `can_proceed == false` | 同上，附缺失字段清单 |
| HITL 审批 | 启用审批策略且填充占比 > `APPROVAL_FILL_RATIO`（0.25） | 只剩 `request_approval`，暂停等人工 |
| 越权拒绝 | Brain 返回白名单外工具 | 拒绝执行，错误进 observation |
| 连续失败熔断 | 连续 3 次工具异常 | 中止，error 终态 |
| 步骤预算 | `max_steps=10` | 强制结束，防原地打转烧 token |

**为什么这比 prompt 约束强**：写在 prompt 里的"请先提取再计算"是建议，模型可能不听；
从白名单里删掉这个工具是**不可能违反**的约束。这是 Agent 工程里最重要的一条思路转换——
**把"希望模型遵守的规则"翻译成"模型看不到的选项"**。

---

## 五、机制三：Brain 双实现（LLM / 规则）

统一协议（Protocol）：

```python
class Brain(Protocol):
    def decide(self, state: dict, available_tools: list[str],
               registry: ToolRegistry) -> BrainDecision: ...
```

`BrainDecision = (tool, args, reason, llm_meta)`

### LLMBrain

- 单次调用，`max_tokens=800`、`temperature=0`；
- 输出强约束 JSON：`{"tool", "args", "reason"}`；
- **解析容错**：先尝试剥离 markdown 代码块围栏，失败则截取第一个 `{` 到最后一个 `}`；
- **自纠重试**：`max_retries=2`，把上一轮的错误原因附进 prompt 引导修正；
- 生产环境经 **LLM 网关** 调用（计量 + 预算 + 熔断）；
- SYSTEM_PROMPT 里明确：`tool` 必须从 `available_tools` 选、禁止编造、reason 面向业务用户、
  finish 时输出 3-6 句中文结论。

### RuleBrain（**本项目最值得抄的一点**）

零 LLM 成本的确定性策略，复刻标准业务路径：
`提取 → 质检门禁 → 建议 → 风险判断 → 填充 → 计算 → 结论`。

三个用途，一个都不能少：

1. **评测**：零 token、逐字节可复现，构成回归基线；
2. **离线 / 无 API Key 场景**：`_select_brain()` 检测不到 Key 就自动降级；
3. **生产兜底**：LLM 网关熔断或预算耗尽时业务不中断。

> 经验：**任何 Agent 系统都应该配一个"关掉 LLM 也能跑通"的规则实现。**
> 它既是成本下限，也是可测试性的来源，还是生产事故的兜底。

选择逻辑（`apps/api/agent_service.py:34`）：
`mode="deterministic"` 或无 `ANTHROPIC_API_KEY` → RuleBrain；否则 LLMBrain（经网关）。

---

## 六、机制四：工具注册表与确定性工具

### 工具定义

```python
Tool(
    name="extract_survey",
    description="从中央空调节能调研表 Excel 提取 12 个月运行数据并做质量检查。返回项目地区、站房类型、数据完整率、置信度和问题列表。",
    input_schema={"type": "object", "properties": {...}, "required": ["survey_path"]},
    handler=extract_survey,      # 控制类工具为 None
    control=False,
)
```

`compact_view()` 给模型看的是 `{name, description, parameters}` —— **description 就是给模型
看的 API 文档**，写得好不好直接决定决策质量。本项目每个 description 都写清了"返回什么"，
让模型能预期观察结果。

### 两条业务线，同一编排器

`build_registry_for_module("air_conditioner" | "air_compressor")`：

| 工具 | 空调线 | 空压线 |
|---|---|---|
| `extract_survey` | 确定性（规则脚本） | **LLM 型**（Word/PPT 非结构化，走 SkillCaller） |
| `suggest_filling` | 确定性（插值→季节值→默认值） | 确定性（附表压力/功率因子/省份电价） |
| `apply_filling` | 确定性（DataFiller，保护字段不可改写） | 确定性 |
| `calculate_saving_rate` | 确定性（`formulas` 跑 Excel 双 sheet 上下限） | 确定性（AirCompressorCalculator） |

### 确定性工具怎么执行 Skill 脚本

```python
def _run_script(script_rel, args) -> str:
    cmd = [sys.executable, str(SKILLS_DIR / script_rel), *args]
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), ...)
```

**零 token、不依赖 API Key、天然没有 Bash/Write 攻击面**——这是 P0 安全改造的核心动作。

### 观察结果裁剪

每个 handler 只返回决策需要的字段（完整率、置信度、条数、前几条问题），
而不是把整个 JSON 塞回去。这是"5 次决策只花 3.5k token"的直接原因。

---

## 七、机制五：控制类工具与 Human-in-the-loop

三个控制类工具**没有 handler**，由编排器拦截并改变运行状态：

| 工具 | 语义 | 终态 |
|---|---|---|
| `finish` | 输出面向用户的结论摘要 | `completed` |
| `request_user_input` | 数据不达标/需补充 | `needs_input` |
| `request_approval` | 填充依赖过重，等人工确认 | `pending_approval` |

**HITL 的本质是"合法终态 + 人工处理后从断点续跑"**，不是"卡在半路"：

1. 白名单判定填充占比 > 0.25 → 只剩 `request_approval`；
2. 运行以 `pending_approval` 结束，`_finalize_run()` 创建待办写入 `agent_actions` 表；
3. 人工在 `/inbox` 审批 → `POST /inbox/{id}/approve`；
4. `resume_agent_run(approval_granted=True)` 从断点续跑，跳过审批门直通。

> 顺序上的坑（已修）：**必须先建待办、再置 stage**。stage 是"本轮结束"的信号，
> 轮询方看到 stage 后要能立刻查到待办，顺序颠倒会产生竞态窗口。

---

## 八、机制六：轨迹可观测性

`agent/trace.py` 记录每步的：工具名、入参、**模型给出的决策理由**、执行结果观察、
耗时（ms）、该次决策的 token 元数据。

```json
{
  "mode": "agent | deterministic",
  "llm_usage": {"brain_calls": 5, "input_tokens": 2966, "output_tokens": 504},
  "steps": [
    {"step": 1, "tool": "extract_survey", "args": {...},
     "reason": "开始提取调研表数据并做质量检查",
     "ok": true, "duration_ms": 4210,
     "observation": {"completeness": 0.92, "confidence_level": "high"},
     "decision": {"model": "...", "input_tokens": 1840, "output_tokens": 96}}
  ],
  "final": {"status": "completed", "summary": "节能率范围为 8.6% - 13.1% ..."}
}
```

- **每步结束即落盘**（`trace.save()`），崩溃也保留已执行部分；
- `on_step` 钩子同步落库 `agent_steps` 表；
- `GET /api/agent/{session_id}/trace` 直接返回；
- 前端 `AgentTraceView.tsx` 逐帧回放"模型在想什么、看到了什么、做了什么"。

> 答辩/演示价值极高：能逐帧展示 Agent 的推理链，而不是只给一个结果。

---

## 九、机制七：持久化、断点恢复与并发锁

### 断点恢复：状态由产物驱动

```python
_STATE_FILES = {
    "extracted_data": "extracted_data.json",
    "quality_report": "quality_report.json",
    "suggestions":    "filling_suggestions.json",
    "filled_data":    "filled_data.json",
    "result":         "result_data.json",
}
```

恢复 = `hydrate_from_session()` 把文件读回 `ctx`，**步号与 token 累计不重置**
（`start_step = 已完成步数 + 1`，`max_steps = 10 + 已完成步数`）。

两个已修复的坑：
- 恢复时必须**重建原 Brain 与工具注册表**，否则空压任务可能按空调工具恢复、
  LLM 模式静默退化成规则模式；
- token 用量**不做二次合并**——每步已由 `add_step()` 实时累计进库，
  `_finalize_run` 再叠加 `trace.llm_usage` 会重复计一倍。

### 并发锁租约（防多副本重复执行）

- `claim_task("agent", session_id, ttl=3600)` 抢占，带 **token**（释放/续租必须校验，防误删他人锁）；
- 后台心跳每 TTL/3 续租一次；续租失败 → `lease.lock_lost` 置位；
- 编排器在**步骤边界**经 `abort_check()` 抛 `LockLostError` **协作式退出**；
- 易主后本实例只标记自己那条 run 为 `interrupted`，**不动会话 stage、不发 done/error 事件**
  （避免与新实例抢 SSE 通道）；
- **先抢锁、再建运行记录**——反序会在抢锁失败时留下 `running` 幽灵记录；
- 线程接管锁之前任何一步失败都必须 `finally` 释放锁，否则遗留一小时锁、重试一直返回"正在执行"。

### 数据表

`sessions` / `agent_runs` / `agent_steps` / `audit_logs` / `llm_calls` /
`agent_actions` / `batch_jobs` / `task_locks`

`LiveSession` 写透：键赋值即落库，业务代码沿用内存写法零改造。

---

## 十、Skill 体系：SKILL.md 与 claude-agent-sdk

### SKILL.md 的标准结构

以 `.claude/skills/extract-and-check-data/SKILL.md` 为例：

```yaml
---
name: extract-and-check-data
description: 从中央空调节能调研表 Excel 中提取月度运行数据,同时进行数据质量检查...
---
```

正文六段：**任务 / 输入参数 / 提取字段（含三级优先级地区推断、月份列对齐、文本解析规则）
/ 质量检查项（必填、范围、逻辑一致性、完整率分级）/ 输出格式（JSON 示例）/ 错误处理**。

> description 是**唯一决定 Skill 是否被正确触发**的字段，必须写清"什么时候用"。

### 12 个 Skill 按性质分流（v2 归宿）

| 类别 | Skill | v2 处置 |
|---|---|---|
| 纯规则脚本 | extract-and-check-data、extract-survey-data、check-data-quality、suggest-data-filling | 脚本**原样保留**，`subprocess` 直调（零 token、零 Key、无 Shell 面）；SKILL.md 退居脚本的行为规格说明 |
| 纯规则脚本（空压/报价/报告） | suggest-air-compressor-data-filling、calculate-air-compressor-energy-saving、generate-air-compressor-report | 同上 |
| 真 LLM 型 | extract-air-compressor-data-llm | **不起 Agent 会话**：本地预提取 + 单次纯文本 LLM 调用（`allowed_tools` 为空）+ 服务端 Schema 校验写盘 |
| 真 LLM 型 | plant-simulation | 保留 SDK 会话（生成并执行 8760h 仿真代码必须 Bash/Write），但**沙箱化** |
| 知识型 | read-energy-survey、read-price-catalog、detect-conflicts | 保留为报价线技能 |

### 安全收敛四招（`core/skill_caller.py`）

1. **`_DETERMINISTIC_SKILLS` 注册表**：4 个纯规则技能直接 subprocess 执行，脱离通用 Agent。
2. **空压提取零 Agent 面**：
   - 文本预提取本地进程内完成；
   - 结构化提取 = 单轮纯文本 LLM 调用，模型只返回 JSON 文本，**全程没有文件系统权限**；
   - 输出经 `_AIR_COMPRESSOR_OUTPUT_SCHEMA` 校验后才由**服务端**写盘；
   - **`_enforce_air_compressor_rejection_rules()` 服务端独立复评打回条件**——
     `shouldReject` 由数据事实决定，**不信模型自报**（曾出现"空项目空设备却自报
     shouldReject=false"绕过打回）。
3. **仿真沙箱**：独立临时目录 cwd、`setting_sources=[]` 隔离、
   Linux/macOS 启用 SDK bash 沙箱并**默认断网**（密钥无法外传）。
4. **初始化不再强依赖 API Key**——确定性路径零 Key 可跑。

**已知边界（如实记录）**：SDK 的 `env` 只能增改不能删除，`ANTHROPIC_API_KEY` 必须保留给 CLI 认证；
Windows 上 SDK 沙箱不可用，退化为 cwd 隔离（生产目标是 Linux 容器）。

---

## 十一、LLM 网关：成本治理五件套

`core/llm_gateway.py`——所有 LLM 调用的统一入口，可直接作为 LLMBrain 的 client
（兼容 `create_message_with_usage` 签名）。

| 能力 | 实现 |
|---|---|
| 计量 | 每次调用落 `llm_calls` 表（谁、什么用途、哪个模型、多少 token、耗时、成败） |
| 预算 | 按用户日 token 预算（默认 50 万），超额抛 `LLMBudgetError` → 上层降级 RuleBrain |
| 熔断 | 连续失败 5 次 → 熔断 60s，避免雪崩式重试烧钱 |
| 重试 | 瞬时错误指数退避重试 2 次（**预算错误不重试**） |
| 路由 | `model_tier` 分 fast / heavy 双模型，决策用便宜模型 |

> 一句话：**LLM 调用不能散落在业务代码里**。收敛到一个网关，才能计量、限流、熔断、降级。

---

## 十二、评测体系：把"正确率"变成可回归的门禁

### 数据集设计（很巧妙）

6 份标注样例调研表放在 `00_sample/示例调研表/`，**文件名自带预期节能率与质量等级**
（如 `【齐全】15.2%惠州光弘`），构成天然锚点；再加 3 个冒烟用例 = 9 例。

### 指标（`evaluation/metrics.py`）

| 指标 | 定义 |
|---|---|
| 流程决策正确率 | 该算的算完、该拦的拦下 |
| 节能率区间偏差 | 到计算区间的距离，`RATE_TOLERANCE_PP = 2.0` |
| **点偏差（区间中点）** | 防"区间越宽越容易准确"的口径作弊 |
| **区间宽度 / 真值覆盖率** | 三项一起看才反映真实精度 |
| 低质量数据中断正确率 | 该拦的是否拦下 |
| 地区识别正确率 | 三级优先级推断是否准 |

### 使用

```bash
python -m evaluation.run_eval                    # 运行并出报告
python -m evaluation.run_eval --save-baseline    # 固化基线
python -m evaluation.run_eval --compare-baseline # 回归检查（有差异退出码 2，CI 门禁）
```

### 诚实原则（本项目做得最好的一点）

报告**不藏分**：点偏差均值 3.56pp、真值覆盖率 0/4 如实写出，并明确"因此不应对外承诺 ≤2%"。
评测的实战战绩：定位出三个真实 bug——月份表头被误读为能耗的静默错数、
深度调研模板变体完整率被低估的门禁误拦、湿球温度 > 干球温度的物理违规数据。
**决策正确率从 83% 修到 100%。**

---

## 十三、平台化工程能力清单

| 能力 | 实现 | 位置 |
|---|---|---|
| SSE 流式事件 | `POST /agent/run/{id}`、`GET /agent/{id}/stream` | `apps/api/routers.py` |
| 轨迹查询 | `GET /agent/{id}/trace` | 同上 |
| 断点续跑 | `POST /agent/run/{id}/resume` | 同上 |
| HITL 收件箱 | `GET /inbox`、`POST /inbox/{id}/approve`、`/dismiss` | 同上 |
| 批量测算 | `POST /agent/batch`、`GET /agent/batch/{id}` | 同上 |
| 事件总线 | 进程内 asyncio.Queue（默认）/ Redis List（多副本，刻意不用 Pub/Sub 因为不缓冲） | `apps/api/events.py` |
| 认证与 ACL | `REQUIRE_AUTH=true` 强制网关身份头；路由级 `acl_guard` 归属校验，越权 403 + 审计 | `apps/api/deps.py` |
| 审计 | 上传/下载/运行/恢复写 `audit_logs` | 同上 |
| 数据库迁移 | Alembic 基线 7 张表；开发走 `create_all`，生产 `alembic upgrade head` | `migrations/` |
| 部署 | Dockerfile + docker-compose（db/api/web）+ CI（lint/单测/评测基线门禁） | `docker-compose.yml` |

---

## 十四、提炼：可复用的 Agent 开发方法论（15 条）

> 这一节是脱离本项目也能用的部分。

**架构与边界**

1. **算得准的交给公式，看得懂的交给模型。** 凡是进报价/进合同/进财务的数字，必须来自确定性引擎。
2. **LLM 只做大脑，不做手脚。** 决策消耗 token，执行必须零 token——把工具实现成确定性函数。
3. **能写规则的环节不要交给 LLM**：规则精确、可单测、可复现、零成本；LLM 反而引入幻觉。

**约束与可靠性**

4. **把"希望模型遵守的规则"翻译成"模型看不到的选项"。** 白名单 > prompt 说教。
5. **硬约束在代码里，软决策在白名单内。** 门禁、依赖、预算、熔断全部代码强制。
6. **永不静默降级。** 计算失败必须显式报错（fail-closed），绝不能返回 mock 假数据——
   假节能率比报错危险一万倍。
7. **永不信任模型自报的判定结果。** 关键判定（如是否打回）由服务端按数据事实独立复评。
8. **给模型最小工具集。** 需要读文件才给 Read，能用纯文本调用解决的就不要起 Agent 会话。

**成本与性能**

9. **观察结果要裁剪。** 回传决策需要的字段，不要塞完整载荷——这直接决定 token 账单。
10. **所有 LLM 调用收敛到一个网关**，才能统一计量、预算、熔断、重试、路由。
11. **配一个"关掉 LLM 也能跑通"的规则实现**，作为评测基线 + 离线兜底 + 生产降级。

**可观测与可测**

12. **每步落盘结构化轨迹**（决策理由 + 观察 + 耗时 + token），崩溃也保留已执行部分。
13. **被拒绝/被熔断的步骤照样记账**，否则 token 口径失真。
14. **评测要防口径作弊**：区间准确率和点偏差、区间宽度、覆盖率一起看。
15. **状态由产物驱动**（而非内存变量），断点恢复、崩溃续跑、多副本才可能成立。

---

## 十五、踩过的坑与修复清单

| # | 问题 | 修复 |
|---|---|---|
| 1 | Excel 计算失败返回 mock 假节能率 | fail-closed，显式抛异常 |
| 2 | 月份表头被误读为能耗（静默错数） | 评测定位后修提取脚本 |
| 3 | 深度调研模板变体完整率被低估 → 门禁误拦 | 修完整率口径 |
| 4 | 湿球温度 > 干球温度的物理违规数据未被拦 | 加逻辑一致性检查 |
| 5 | token 用量在 finalize 时二次合并 → 计两倍 | 每步 `add_step()` 实时累计，finalize 只更新状态 |
| 6 | 先建 run 记录再抢锁 → 抢锁失败留 running 幽灵记录 | 先抢锁再建记录 |
| 7 | 抢锁后到线程接管前失败 → 遗留一小时锁 | `except BaseException` 释放，worker `finally` 释放 |
| 8 | 断点恢复不重建 Brain/Registry → 空压按空调工具恢复 | 恢复时按原 mode 与 module_type 重建 |
| 9 | 锁易主后两个实例并发写同一会话 | 心跳续租 + 步骤边界 `abort_check` 协作式退出 |
| 10 | HITL 待办与 stage 更新顺序颠倒 → 竞态窗口 | 先建待办再置 stage |
| 11 | 空压提取"空项目空设备却自报 shouldReject=false" | 服务端独立复评打回规则 |
| 12 | 模型越权调用白名单外工具 | 拒绝执行 + 错误进 observation（但仍落库记账） |
| 13 | `formulas` 库忽略 `calculate(inputs=...)` | 先写文件再加载计算 |
| 14 | 三份 `SkillCaller` 拷贝各自演化 | 收敛为统一通道 |

---

## 十六、命令速查与代码索引

```bash
# 启动（Streamlit 遗留入口）
streamlit run app.py

# 测试（全量 314 passed / 25 skipped）
PYTHONPATH=$PWD pytest
pytest tests/unit/test_agent_orchestrator.py -v -s

# 评测（零 token，可复现）
python -m evaluation.run_eval
python -m evaluation.run_eval --save-baseline
python -m evaluation.run_eval --compare-baseline   # 有差异退出码 2，CI 门禁

# 数据库迁移
alembic upgrade head

# 会话目录
tmp/session_YYYYMMDD_HHMMSS/   # upload_survey.xlsx / *.json / calculation_result.xlsx / agent_trace.json
```

**关键代码入口**

| 我想看… | 去这里 |
|---|---|
| 主循环怎么跑 | `agent/orchestrator.py::run` |
| 白名单怎么算 | `agent/orchestrator.py::_allowed_tools` |
| 模型看到什么 | `agent/orchestrator.py::_state_view`、`agent/brain.py::_build_prompt` |
| JSON 决策怎么解析容错 | `agent/brain.py::LLMBrain._parse_decision` |
| 工具怎么定义 | `agent/tools.py::build_registry_for_module` |
| 确定性脚本怎么调 | `agent/tools.py::_run_script` |
| 服务层怎么选 Brain / 起线程 | `apps/api/agent_service.py::_select_brain`、`_start_run_thread` |
| SSE 端点 | `apps/api/routers.py`（/agent/run、/stream、/trace、/resume、/inbox、/batch） |
| 安全收敛 | `core/skill_caller.py::_DETERMINISTIC_SKILLS`、`_validate_air_compressor_payload`、`_stage_sandbox` |
| 评测判定 | `evaluation/metrics.py::judge_case` |

**遗留边界（下一步）**

- 冷站仿真 / 报价线尚未接入统一 orchestrator（注册为 LLM 型工具即可）；
- Windows 上 SDK 沙箱不可用，仅 cwd 隔离；
- `INDUSTRY_DEFAULTS` 行业默认值需业务校准（深度模板变体偏差 5.42pp）；
  点偏差覆盖率 0/4 需 30-50 例专家签字校准集；
- SSE 队列 / LiveSession 写缓存为进程内，多副本需 Redis 模式 + 网关粘滞路由。
