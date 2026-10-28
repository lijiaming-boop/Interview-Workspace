#!/usr/bin/env bash
# 部署冒烟验证：网关健康 → 未登录拦截 → 无头 OAuth 登录 → 业务链路（上传 + 确定性测算）
#
# 用法: ./scripts/verify_deployment.sh [BASE_URL]
#   BASE_URL 默认 http://localhost:8080；必须与 .env 中 REDIRECT_URI 同主机，
#   否则登录回调写入的会话 Cookie 不会随后续请求发送。
# 依赖: curl、python3
set -uo pipefail
BASE="${1:-http://localhost:8080}"
TEST_USER="${TEST_USER:-test}"
TEST_PASS="${TEST_PASS:-test123456}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# 相对路径 + cd 仓库根：Windows(Git Bash) 的 curl 读不了含空格的 MSYS 绝对路径
cd "$REPO" || exit 1
SURVEY="${SURVEY_FILE:-./00_sample/示例调研表/【参差不齐】11.9%中央空调节能调研表（首页含问题指引）.xlsx}"

PASS=0; FAIL=0
ok()   { echo "  ✅ $*"; PASS=$((PASS+1)); }
bad()  { echo "  ❌ $*"; FAIL=$((FAIL+1)); }
step() { echo; echo "== $* =="; }
jget() { python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get(sys.argv[1],""))' "$1" 2>/dev/null; }

JAR="$(mktemp)"; PG="$(mktemp)"
trap 'rm -f "$JAR" "$PG"' EXIT

step "1/5 网关健康 $BASE/health"
body="$(curl -s --max-time 10 "$BASE/health" || true)"
[ "$body" = '{"status":"ok"}' ] && ok "proxy /health 200" || bad "proxy /health 响应异常: $body"

step "2/5 未登录拦截"
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE/" || true)"
loc="$(curl -s -o /dev/null -w '%{redirect_url}' --max-time 10 "$BASE/" || true)"
if [ "$code" = "307" ] && [[ "$loc" == */login ]]; then
  ok "未登录访问 / → 307 /login"
else
  bad "期望 307→/login，实际 $code→$loc"
fi
auth_url="$(curl -s -o /dev/null -w '%{redirect_url}' --max-time 15 -c "$JAR" -b "$JAR" "$BASE/login" || true)"
case "$auth_url" in
  *"response_type=code"*|*"/protocol/openid-connect/auth"*) ok "/login → OIDC 授权端点" ;;
  *) bad "/login 未跳转 OIDC 授权端点: $auth_url" ;;
esac

step "3/5 无头 OAuth 登录（$TEST_USER）"
curl -s --max-time 20 -c "$JAR" -b "$JAR" "$auth_url" > "$PG" || true
action="$(python3 -c '
import re, sys, html
page = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
m = re.search(r"<form[^>]*action=\"([^\"]+)\"", page)
print(html.unescape(m.group(1)) if m else "")' "$PG" 2>/dev/null || true)"
if [ -z "$action" ]; then
  bad "未解析到 Keycloak 登录表单（IdP 页面异常？）"
  echo "----- 登录页前 20 行 -----"; head -20 "$PG"
  exit 1
fi
# 提交登录表单并跟随回调重定向（302 后 curl 自动转 GET，不能加 -X POST——
# 它会强制重定向链上所有请求都用 POST，导致 /oauth2callback 收到 POST 而 404）
curl -s -o /dev/null --max-time 60 -L --max-redirs 8 \
  -c "$JAR" -b "$JAR" \
  --data-urlencode "username=$TEST_USER" \
  --data-urlencode "password=$TEST_PASS" \
  --data-urlencode "credentialId=" \
  --data-urlencode "login=Sign In" \
  "$action" || true
if grep -q 'session_token' "$JAR"; then
  ok "获得网关会话 Cookie（session_token）"
else
  bad "未获得会话 Cookie（账号密码错误或 IdP 配置问题）"
  exit 1
fi
app_page="$(curl -s --max-time 20 -c "$JAR" -b "$JAR" "$BASE/" || true)"
printf '%s' "$app_page" | grep -q "节能测算" \
  && ok "应用首页渲染（Next.js 前端可达）" \
  || bad "应用首页异常（无预期业务标记）"

step "4/5 业务链路：api 健康 + 上传调研表"
h="$(curl -s --max-time 15 -c "$JAR" -b "$JAR" "$BASE/api/health" || true)"
printf '%s' "$h" | grep -q '"status":"ok"' \
  && ok "/api/health 经 proxy→web(BFF)→api 全链路可达" \
  || bad "/api/health 异常: $h"
if [ ! -f "$SURVEY" ]; then
  bad "样例调研表不存在: $SURVEY（可用 SURVEY_FILE=... 指定，跳过测算验证）"
else
  up="$(curl -s --max-time 120 -c "$JAR" -b "$JAR" \
        -F "file=@$SURVEY;filename=survey.xlsx" \
        "$BASE/api/upload?module_type=air_conditioner" || true)"
  sid="$(printf '%s' "$up" | jget session_id)"
  if [ -n "$sid" ]; then
    ok "上传成功 session_id=$sid"
  else
    bad "上传失败: $up"
    exit 1
  fi

  step "5/5 确定性 Agent 测算（零 LLM 成本）"
  run="$(curl -s --max-time 30 -c "$JAR" -b "$JAR" \
         -X POST "$BASE/api/agent/run/$sid?mode=deterministic" || true)"
  [ "$(printf '%s' "$run" | jget status)" = "running" ] \
    && ok "agent 已启动（deterministic 模式）" \
    || { bad "agent 启动失败: $run"; exit 1; }

  status=""
  for _ in $(seq 1 100); do
    sleep 3
    status="$(curl -s --max-time 15 -c "$JAR" -b "$JAR" "$BASE/api/agent/$sid/trace" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    print((d.get("final") or {}).get("status") or d.get("status") or "")
except Exception:
    pass' 2>/dev/null || true)"
    case "$status" in
      completed|error|needs_input|pending_approval) break ;;
    esac
  done
  case "$status" in
    completed) ok "测算完成（completed），节能率结论已产出" ;;
    needs_input|pending_approval)
      bad "测算被门禁拦截（status=$status）——样例数据预期可通过，请查 docker compose logs api" ;;
    *) bad "测算未在期限内完成（最后 status=${status:-<空>}）" ;;
  esac
fi

echo
echo "=========================================="
echo "  通过 $PASS / $((PASS+FAIL))"
echo "=========================================="
[ "$FAIL" = "0" ]
