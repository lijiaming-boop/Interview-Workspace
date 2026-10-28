# 从 Skill 到 Agent 的升级改造设计

> 本文档记录 HVAC Energy Agent 的 AI 架构升级全过程：从最初的
> **claude-agent-sdk + Skill 脚本调用**（v1，"LLM 混合架构"），
> 升级为**自研 AgentOrchestrator 统一编排**（v2，agent/ 包）的
> 动机、设计决策、迁移映射、Skill 体系归宿、安全加固与效果验证。
>
> 配套：[v1 技术方案（03_plan）](./03_plan/技术方案详细设计-LLM混合架构.md) ·
> [ARCHITECTURE（v2 决策边界）](./ARCHITECTURE.md) ·
> [改造文档（平台化全程）](./改造文档.md) · [评测报告](./evaluation/report.md)
>
> 最后更新：2026-08-30

---

## 一、结论速览（TL;DR）

v1 把"每一步业务动作"都包装成一个 Skill，由 `SkillCaller` 为每个动作起一个
**完整的 claude-agent-sdk Agent 会话**（带 Bash/Read/Write 工具）去执行；
流程顺序写死在 Streamlit/服务层的状态机里。v2 把这套东西**升级为一个进程内的
ReAct 编排器**：业务能力变成带 JSON Schema 的**确定性工具**（执行零 token），
"决定下一步做什么"收敛为**单次 JSON 决策调用**（LLMBrain/RuleBrain 双实现），
硬约束（质量门禁/依赖/越权）由**工具白名单**在代码层强制，全程产生结构化轨迹。

| 维度 | v1：claude-agent-sdk + Skill | v2：AgentOrchestrator |
|---|---|---|
| LLM 调用粒度 | 每个业务阶段起一个完整 Agent 会话（含工具循环） | 每步一次小额 JSON 决策调用；工具执行零 token |
| 一次完整测算成本 | 3 个会话（提取/质检/建议）+ LLM 增强，数十次内部模型调用 | **5 次决策调用 ≈ 3.5k tokens**（input 2,966 / output 504） |
| 流程编排 | 写死的五阶段状态机（UI/服务层 if-else） | ReAct 循环 + 白名单门禁，模型在约束内自主决策 |
| 硬约束 | 无（会话内模型可用 Bash/Write 自由行动） | 白名单前置依赖、质量门禁、熔断、步预算，**模型绕不过** |
| 可观测性 | ai_logs 文本流 | `agent_trace.json` 结构化轨迹（决策理由/观察/耗时/token）+ 落库 + SSE |
| 可测试性 | 起会话依赖 API Key，只能 mock | RuleBrain 零成本复现；评测基线 9 例回归门禁 |
| 决策正确率 | 无法度量 | **100%**（9/9 评测用例） |
| 持久化/恢复 | 进程内 dict 会话 | SQL 落库（agent_runs/steps）+ 断点恢复 + HITL 暂停续跑 |
| claude-agent-sdk 去向 | 一切 LLM 动作的唯一通道 | **收缩为 LLM 型工具**（空压提取、冷站仿真）的执行底座 |

**一句话**：不是推倒重来——v1 沉淀的 Skill 纯 Python 脚本、Excel 公式引擎、
会话产物协议被原样保留；升级的是**调度方式**（会话 → 工具）与**控制权**
（状态机 → 白名单内的 ReAct 自主决策）。

---

## 二、v1 回顾：claude-agent-sdk + Skill 架构

### 2.1 演变轨迹

1. **2025-12-17 v1 技术方案**（`03_plan/技术方案详细设计-LLM混合架构.md`）：
   Streamlit UI + 三个 LLM Skills（extract-survey-data / check-data-quality /
   suggest-data-filling）+ Excel 公式引擎，Skill 最初设计为
   `subprocess` 调 Claude CLI。
2. **2025-12-20~21 MVP 落地**（`docs/IMPLEMENTATION_REPORT.md`）：
   `core/skill_caller.py`（SkillCaller）+ 三个 Skill + Streamlit 六阶段 UI。
3. **2025-12-24 迁移到 SDK**（`docs/plans/2025-12-24-migrate-skill-caller-to-agent-sdk.md`）：
   CLI subprocess 方式改为 `claude_agent_sdk.query()` 异步调用，
   Skills 由 SDK 自动发现（此后陆续演进为 12 个 Skill）。
4. **2025-12-26 计算引擎定型**：xlwings → `formulas` 库纯 Python 跑 Excel 公式。
5. **2026-07-22~23 Web 化重构**（refactor/nextjs-fastapi 分支）：Next.js + FastAPI
   取代 Streamlit 单体，所有 LLM 调用**进一步收拢到 claude-agent-sdk**
   （commit 21ef5d4 "route all LLM calls through claude-agent-sdk"），
   空压/仿真/报价三条业务线接入同一 SkillCaller 模式。

### 2.2 工作方式

```
Streamlit / Next.js 状态机（stage: upload → extract_check → filling → calculate → result）
        │ 每阶段一次
SkillCaller（core/skill_caller.py）
        │ claude_agent_sdk.query() —— 一个完整 Agent 会话
        │ 模型加载 .claude/skills/<name>/SKILL.md 指令
        │ 模型自主使用 Read/Write/Bash 工具
        ▼
会话目录 JSON 文件交换（extracted_data.json → quality_report.json → …）
        │
formulas Excel 公式引擎（复制模板 → 回填 → 计算 → 提结果）
```

要点：**LLM 不只是"理解"，还负责"执行"**——模型在会话里自己读 Excel、
跑脚本、写输出文件；每次调用 300~900 秒超时，支持多轮重试
（commit 370bed5：`max_retries=2`，失败后让 Agent 自行修复重跑）。

### 2.3 v1 的结构性问题（升级动机）

| # | 问题 | 具体表现 |
|---|---|---|
| 1 | **为"执行一条 python 命令"付整个会话的钱** | 提取/质检/建议三个环节早已是纯规则 Python 脚本（v1 后期演进为规则实现），但仍经 SkillCaller 起完整 Agent 会话——SKILL.md 重新加载、工具集重新展开、模型逐步自行操作文件，token 与延迟全部浪费在"搬运"上 |
| 2 | **编排写死、不可演进** | 流程顺序硬编码在 `app.py`/`services.py` 的五阶段状态机 if-else 里；换流程=改代码；无法让模型根据观察结果动态决定路径（如数据差时提前中断、求助） |
| 3 | **行为不可证明、安全面失控** | 会话内模型持有 Bash/Write（空压提取甚至 bypassPermissions），处理用户上传文件等于把任意代码执行暴露给不可信输入；流程正确性依赖"模型自觉" |
| 4 | **不可观测** | 只有 ai_logs 文本流；无结构化轨迹——无法回答"模型当时为什么这么做、每步花了多少 token" |
| 5 | **不可测试** | 三份 SkillCaller 拷贝（`core/`、`quote_module/`、`core/quote/`）各自演化；起会话强依赖 API Key（无 Key 直接 SkillError），单测全靠 mock；16 个既有测试长期失败无人修 |
| 6 | **无持久化、无恢复** | 会话是进程内 dict，重启即丢；失败静默降级 mock 假数据（Excel 计算失败返回假节能率） |

---

## 三、v2 设计目标与原则

1. **边界原则**：算得准的交给公式（确定性工具，执行零 token），看得懂的交给模型
   （感知/决策/叙事）。
2. **硬约束在代码里，软决策在白名单内**：门禁、依赖、预算由编排器强制；
   模型的自主性被限制在白名单可选空间内，且可被代码证明。
3. **可评测、可复现**：同一编排器跑确定性策略时零 token、逐字节可复现，
   构成回归门禁。
4. **复用而非重写**：v1 的规则脚本、Excel 引擎、JSON 产物协议原样保留，
   只换调度层。

---

## 四、核心改造映射（v1 → v2）

| v1 概念 | v2 概念 | 位置 |
|---|---|---|
| `.claude/skills/<name>/SKILL.md` + 纯 Python 脚本 | **Tool**（JSON Schema 描述 + 确定性 handler，直接 subprocess 执行同一脚本） | `agent/tools.py` |
| `SkillCaller`（每动作一个 claude-agent-sdk 会话） | **两个去处**：确定性技能 → 直接执行脚本；真正的决策 → **Brain**（单次 JSON 调用，经 LLM 网关） | `agent/brain.py`、`core/llm_gateway.py` |
| 五阶段状态机（UI/服务层写死） | **AgentOrchestrator**（ReAct 循环：决策→执行→观察→再决策） | `agent/orchestrator.py` |
| 状态机 if-else 里的"下一步" | **RuleBrain**（同一 Brain 接口的确定性策略，评测/离线/无 Key 场景） | `agent/brain.py` |
| ai_logs 文本流 | **AgentTrace** 结构化轨迹（每步落盘 `agent_trace.json` + `agent_runs/steps` 落库 + SSE 推流） | `agent/trace.py` |
| 进程内 dict 会话 | **PersistentSessionStore**（LiveSession 写透落库）+ 断点恢复（hydrate） | `apps/api/storage.py`、`agent/orchestrator.py` |
| Excel 失败静默 mock 假数据 | 计算失败**显式报错**（fail-closed） | `core/excel_calculator.py` |
| 单一空调流程 | **双业务线统一编排**：`build_registry_for_module("air_conditioner" / "air_compressor")`，同一编排器与门禁语义，不同工具集 | `agent/tools.py` |

关键语义变化：v1 中 LLM **既是大脑又是手脚**（在会话内自主使用工具完成任务）；
v2 中 LLM **只是大脑**（选择下一个工具并给出理由），手脚全部是确定性函数——
token 只花在决策点，执行零 token。

---

## 五、v2 架构详解（agent/ 四模块）

### 5.1 AgentOrchestrator（`agent/orchestrator.py`）

ReAct 主循环，每步依次：`abort_check`（锁租约协作式中止）→ `_allowed_tools()`
计算白名单 → `brain.decide(state, allowed, registry)` → 越权校验 →
执行（确定性工具）或拦截（控制类工具）→ 记录轨迹/落库 → 循环。

**白名单硬约束**（`_allowed_tools`，模型无法绕过）：

| 约束 | 触发条件 | 强制行为 |
|---|---|---|
| 工具前置依赖 | 未提取/未建议/未填充 | 白名单中不出现对应下游工具 |
| 质量门禁 | 数据置信度 low（完整率 <50%） | 白名单只剩 `request_user_input`，流程以 needs_input 终止 |
| 站房类型拦截 | 提取判定 unsupported（热水泵/蓄冷/中低温机组） | 同上，附识别依据 |
| 不可填充拦截 | 规则引擎返回 `can_proceed=false` | 同上，附缺失字段清单 |
| HITL 审批 | 启用审批策略且填充占比 > `APPROVAL_FILL_RATIO`（0.25） | 白名单只剩 `request_approval`，暂停等人工；批准后 `approval_granted=True` 续跑直通 |
| 越权拒绝 | Brain 返回白名单外工具 | 拒绝执行，错误作为观察结果反馈（防幻觉/越权），该步照样落库（token 口径不丢） |
| 连续失败熔断 | 同一运行连续 3 次工具异常 | 中止，error 终态 |
| 步骤预算 | `max_steps=10` | 强制结束，防止原地打转烧 token |

**断点恢复**：每个确定性工具把产物写为标准文件名（extracted_data.json 等），
状态完全由产物驱动——恢复 = `hydrate_from_session()` 把文件读回 ctx，
步号与 token 累计不重置。HITL 暂停（pending_approval/needs_input）本质是
"合法终态 + 人工处理后从断点续跑"。

### 5.2 Brain 双实现（`agent/brain.py`）

统一协议 `decide(state, available_tools, registry) -> BrainDecision(tool, args, reason)`：

- **LLMBrain**：单次 LLM 调用（`max_tokens=800`、temperature=0），输出强约束 JSON
  `{"tool", "args", "reason"}`；解析失败剥离 markdown 围栏重试（max_retries=2），
  上一轮错误附进 prompt 引导修正；越出 `available_tools` 直接判非法。
  生产环境经 **LLM 网关**（`core/llm_gateway.py`）调用——计量落库、按用户日预算
  超额拒绝、连续失败熔断、退避重试、fast/heavy 双模型路由。
- **RuleBrain**：零 LLM 成本的确定性策略，复刻标准业务路径
  （提取→质检门禁→建议→风险判断→填充→计算→结论），`RISKY_FILL_RATIO=0.5`
  判断填充依赖过重转人工。用于评测（逐字节可复现）、离线运行与无 API Key 场景。

两种 Brain 与编排器解耦——选哪种由 `apps/api/agent_service.py::_select_brain`
按环境决定（mode=deterministic 或未配 Key → RuleBrain）。

### 5.3 ToolRegistry（`agent/tools.py`）

每个工具 = 名称 + 面向决策模型的描述 + JSON Schema + 确定性 handler。
控制类工具（`finish` / `request_user_input` / `request_approval`）无 handler，
由编排器拦截改变运行状态。

- **空调线（全确定性）**：`extract_survey`（合并原 extract+check 两个 skill）、
  `suggest_filling`、`apply_filling`、`calculate_saving_rate`。
- **空压线（M2 统一编排）**：`extract_survey` 为 **LLM 型工具**（Word/PPT 非结构化
  文本必须语言模型，经 SkillCaller 通道），suggest/apply/calculate 为确定性工具。
- **共享策略函数** `fill_ratio()`：填充占比的分母按模板可用字段计算
  （模板天生没有的字段不算质量风险）。

### 5.4 AgentTrace（`agent/trace.py`）

记录每步的工具、参数、模型给出的决策理由、观察结果、耗时与该次决策的
token 元数据；每步结束即落盘 `agent_trace.json`（崩溃也保留已执行部分），
`on_step` 钩子同步落库。`GET /api/agent/{id}/trace` 直接返回，Next.js
`/agent` 页面逐帧回放"模型在想什么、看到了什么、做了什么"。

---

## 六、Skill 体系的归宿：保留、改造、收敛

升级**没有废弃** `.claude/skills/`——12 个 Skill 按性质分流：

| 类别 | Skill | v2 处置 |
|---|---|---|
| 纯规则脚本（空调线） | extract-and-check-data、extract-survey-data、check-data-quality、suggest-data-filling | **脚本原样保留**，由 `agent/tools.py::_run_script` 直接 subprocess 执行（零 token、零 Key、无 Shell 面）；SKILL.md 退居脚本的行为规格说明 |
| 纯规则脚本（空压/报价/报告） | suggest-air-compressor-data-filling、calculate-air-compressor-energy-saving、generate-air-compressor-report 等 | 同上：确定性技能改为直接执行受控脚本（P0 安全改造，见第七节） |
| 真 LLM 型 | extract-air-compressor-data-llm | 仍由 SkillCaller 通道服务，但 **P0 后不再起 Agent 会话**：本地预提取 + 单次纯文本 LLM 调用（allowed_tools 为空），输出经服务端 JSON Schema 校验后写盘——模型全程没有文件系统权限 |
| 真 LLM 型 | plant-simulation | 保留 claude-agent-sdk 会话（生成并执行 8760h 仿真代码必须 Bash/Write），但沙箱化：独立临时目录 cwd、`setting_sources=[]` 隔离、SDK bash 沙箱断网（Linux/macOS） |
| 知识型 | read-energy-survey、read-price-catalog、detect-conflicts | 保留为报价线技能（报价线尚未接入统一编排器，见第十节） |

原 v1 提示词资产的转移路径：SKILL.md 中的**业务判定规则**（打回条件、
模板约束、报告规范）下沉为脚本代码与服务端校验；**流程知识**（下一步该做什么）
转移到 `brain.py` 的决策协议（LLMBrain SYSTEM_PROMPT / RuleBrain 规则序列）；
仅生成类任务（仿真代码）仍由提示词在沙箱内驱动模型。

claude-agent-sdk 的最终定位：**LLM 型工具的执行底座**，而非流程编排层——
编排主权上收到 AgentOrchestrator。

---

## 七、P0 安全加固对 Agent 面的收敛（2026-08-30）

升级后残余的 claude-agent-sdk 使用面经《改进方案1》安全改造进一步收敛
（`core/skill_caller.py`）：

1. **确定性技能脱离通用 Agent**：`_DETERMINISTIC_SKILLS` 注册表内的技能直接
   subprocess 执行脚本——不消耗 token、不依赖 API Key、天然没有 Bash/Write 攻击面。
2. **空压提取零 Agent 面**：文本预提取本地进程内完成，结构化提取为单轮纯文本
   LLM 调用；模型输出经 `_AIR_COMPRESSOR_OUTPUT_SCHEMA` 校验后由**服务端**写盘；
   打回判定（`_enforce_air_compressor_rejection_rules`）由服务端按 SKILL.md
   条件独立复评——`shouldReject` 由数据事实决定，不信模型自报。
3. **仿真沙箱**：plant-simulation 每次调用在独立沙箱目录执行，不加载宿主项目
   设置，Linux/macOS 启用 SDK bash 沙箱并默认断网（密钥无法外传）。
4. `SkillCaller` 初始化不再强依赖 API Key（确定性路径零 Key 可跑）。

---

## 八、迁移与兼容策略

- **端点并存**：v1 五阶段端点作为兼容端点保留，`POST /api/agent/run`（SSE）
  为新主线；批量测算 `/api/agent/batch` 复用同一编排器。
- **产物协议不变**：会话目录内 extracted_data.json / quality_report.json /
  filling_suggestions.json / filled_data.json / result_data.json 的文件名与结构
  与 v1 完全一致——它同时是工具间协议、断点恢复数据源与旧前端兼容层。
- **业务零改造接持久化**：`LiveSession` 写透存储让旧代码"键赋值即落库"，
  内存写法不变。
- **模块 facade**：路由拆分保持 `from apps.api.routers import router` 兼容。

---

## 九、效果与验证

| 指标 | 结果 | 验证方式 |
|---|---|---|
| LLM 成本 | 一次完整测算 5 次决策调用 ≈ 3.5k tokens（input 2,966 / output 504）；评测/离线模式 0 token | `agent_trace.json.llm_usage`；`GET /api/llm/usage` 计量 |
| 流程决策正确率 | 100%（9/9：6 锚点 + 3 冒烟用例） | `python -m evaluation.run_eval --compare-baseline`（差异退出码 2，CI 门禁） |
| 中断正确率 / 地区识别 | 2/2 / 100% | 同上 |
| 节能率偏差 | 区间口径平均 1.97pp；点偏差均值 3.56pp、真值覆盖率 0/4（诚实口径，佐证不应对外承诺 ≤2%） | 评测报告 |
| 测试 | 314 passed / 25 skipped（unit+integration+e2e+quote，排除 4 个需真实 LLM 的用例） | pytest 命令见改造文档第六节 |
| 可观测 | 全量轨迹可逐帧回放（Next.js /agent 页） | `GET /api/agent/{id}/trace` |

评测的实战战绩（评测驱动修复的三个真实问题）：月份表头被误读为能耗的静默错数
bug、深度调研模板变体完整率被低估的门禁误拦、湿球温度 > 干球温度的物理违规数据——
决策正确率从 83% 修到 100%。

---

## 十、遗留边界与下一步

| 项 | 说明 | 去向 |
|---|---|---|
| 冷站仿真/报价线未接入统一 orchestrator | 两者已有独立 agent 面；仿真已沙箱化，报价仍用独立流水线 + detect-conflicts 技能 | 注册为 LLM 型工具即可，路线图 |
| Windows 上 SDK 沙箱不可用 | 仿真沙箱在 Windows 退化为 cwd 隔离；生产部署目标为 Linux 容器 | 已知边界 |
| 行业默认值参数校准 | 深度模板变体偏差 5.42pp，需业务校准 `INDUSTRY_DEFAULTS`；点偏差覆盖率 0/4 必须靠 30-50 例专家签字校准集解决 | 业务确认 |
| SSE 事件/LiveSession 写缓存为进程内 | 多副本需网关按 session 粘滞路由；事件总线已有 Redis List 缓冲模式（`apps/api/events.py`） | 多副本部署时启用 |

---

## 附：演进时间线（git 提交索引）

| 日期 | 提交 | 里程碑 |
|---|---|---|
| 2025-12-17 | — | v1 技术方案（LLM 混合架构，CLI subprocess 设计） |
| 2025-12-20~21 | 7f4c0d7 等 | MVP：SkillCaller + 三 Skill + Streamlit 六阶段 UI（`docs/IMPLEMENTATION_REPORT.md`） |
| 2025-12-24 | — | SkillCaller 迁移到 claude-agent-sdk（docs/plans/2025-12-24-…） |
| 2025-12-26 | — | xlwings → formulas 纯 Python Excel 引擎 |
| 2026-07-22 | 26a1f03 等 | Next.js + FastAPI 重构起步（仿真模块迁移）；2026-08-03 合并主干 |
| 2026-07-23 | 21ef5d4 / 624ed14 | 全部 LLM 调用收拢经 claude-agent-sdk（v1 终态） |
| 2026-08-29 | **c307ade** | **Agent 化核心改造（P1-P3）：agent/ 四模块 + evaluation/ + 文档重写** |
| 2026-08-29 | cf09523 | 16 个既有测试修复 + 深度模板变体适配（评测 100%） |
| 2026-08-29 | cf66e95 | M1 平台底座：落库/认证 ACL/断点恢复/Docker/CI |
| 2026-08-29 | 7f9c049 | M2/M3：LLM 网关/HITL/空压统一编排/批量/历史库/通知 |
| 2026-08-30 | bdf4605 | 事件总线（Redis List）+ Alembic 基线 + 前端 Agent 控制台 |
| 2026-08-30 | — | P0/P1 安全与正确性改造（改进方案1）：确定性技能脱 Agent、LLM 技能最小工具集、仿真沙箱 |
