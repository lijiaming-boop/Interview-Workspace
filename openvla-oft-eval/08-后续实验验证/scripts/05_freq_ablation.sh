#!/usr/bin/env bash
# =============================================================================
# 05_freq_ablation.sh —— 仿真侧 · 高频控制 20Hz vs 50Hz
#
# 为什么这样做：
#   OFT 一次并行解码整条 chunk，算力允许把 LIBERO 控制器频率提到 50Hz，
#   在本实验中固定 open-loop=4，使单个窗口从 200ms(4步@20Hz) 缩短到 80ms(4步@50Hz)，
#   减少开环漂移，
#   预期缓解 230/233 的“卡住/空抓后漂移”。
#   已知风险：checkpoint 在 20Hz 动作分布上训练，50Hz 存在控制分布偏移，
#   成功率可能不升反降 —— 所以必须同时观测 deadline（推断延迟是否跟上）。
# 实施：--control_freq 20/50；runner 会按频率缩放 max_steps 与 wait_steps，
#   并在 chunks.jsonl 记录 chunk_deadline_ms / chunk_deadline_met。
# 用法：bash 05_freq_ablation.sh [freq20|freq50|all]
# 验收：
#   - freq50 的 chunk_deadline_met≥95%，且 P95 推断延迟≤80ms，才是有意义的 50Hz；
#   - 若 deadline 达标但成功率明显下降 → 证实分布偏移，需 50Hz 数据重训（转 07）；
#   - 若失败模式消失且成功率不降 → 直接采用 50Hz。
# =============================================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_SCRIPT="${SELF_DIR}/verify_frequency_ablation.py"
source "${SELF_DIR}/_common.sh"
require_full_patch; require_checkpoint; require_env

MODE="${1:-all}"
case "${MODE}" in
  freq20) study_run_eval "sim-frequency-20hz" "4:32,6:20,7:27" 4 1.0 False True 0.0 20 ;;
  freq50) study_run_eval "sim-frequency-50hz" "4:32,6:20,7:27" 4 1.0 False True 0.0 50 ;;
  all)
    study_run_eval "sim-frequency-20hz" "4:32,6:20,7:27" 4 1.0 False True 0.0 20
    study_run_eval "sim-frequency-50hz" "4:32,6:20,7:27" 4 1.0 False True 0.0 50
    python "${VERIFY_SCRIPT}" \
      --freq20-root "${RESULT_BASE}/sim-frequency-20hz" \
      --freq50-root "${RESULT_BASE}/sim-frequency-50hz"
    ;;
  *) echo "Usage: $0 {freq20|freq50|all}" >&2; exit 2 ;;
esac

if [ "${MODE}" != "all" ]; then
  echo "单组运行完成。两组都完成后执行："
  echo "python ${VERIFY_SCRIPT} --freq20-root ${RESULT_BASE}/sim-frequency-20hz --freq50-root ${RESULT_BASE}/sim-frequency-50hz"
fi
