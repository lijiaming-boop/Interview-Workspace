#!/usr/bin/env bash
# HVAC Energy Agent 一键部署（Linux 服务器，含本地 Keycloak 测试 IdP）
#
# 用法:
#   ./scripts/deploy_server.sh [PUBLIC_BASE_URL]
#   例: ./scripts/deploy_server.sh http://192.168.1.100
#
# 步骤: 补全 .env → 渲染 Keycloak 测试域 → compose 构建并启动
#       (db + redis + api + web + proxy + oidc) → 等待就绪 → 冒烟验证
# 前置: Docker 24+ 与 Compose v2 插件；8080/8081 端口可用；
#       PUBLIC_BASE_URL 需浏览器与容器都可达（用服务器 IP 即可）。
set -euo pipefail
cd "$(dirname "$0")/.."

log()  { echo ">> $*"; }
die()  { echo "❌ $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "缺少命令: $1"; }

need docker
need curl
need sed
docker compose version >/dev/null 2>&1 || die "需要 Docker Compose v2（docker compose 插件）"

# ---------- 1. 服务器地址 ----------
PUB="${1:-}"
if [ -z "$PUB" ]; then
  default_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [ -z "$default_ip" ]; then
    default_ip="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')"
  fi
  printf "浏览器访问地址（回车默认 http://%s）: " "${default_ip:-<服务器IP>}"
  read -r PUB
  PUB="${PUB:-http://$default_ip}"
fi
PUB="${PUB%/}"
case "$PUB" in http://*|https://*) : ;; *) PUB="http://$PUB" ;; esac
# IdP 宿主机端口可配置（默认 8081；若被占用用 OIDC_PORT=xxx 覆盖）
OIDC_PORT="${OIDC_PORT:-8081}"
REDIRECT_URI="$PUB:8080/oauth2callback"
OAUTH_SERVER_URL="$PUB:$OIDC_PORT"
log "PUBLIC_BASE_URL  = $PUB"
log "回调地址         = $REDIRECT_URI"
log "OIDC 发现地址    = $OAUTH_SERVER_URL"

# ---------- hairpin 修复：云服务器 VM/容器访问自身公网 IP 常被 NAT 拦截 ----------
# proxy(容器内) 启动时要连 OAUTH_SERVER_URL 做 OIDC discovery；部署脚本冒烟验证
# 也从宿主机访问 PUB。若 VM 无法回环访问自身公网 IP（腾讯云/阿里云典型行为），
# 用 iptables DNAT 把公网 IP 的发布端口转回内网 IP（幂等；重启后需重跑本脚本）。
pubhost="${PUB#http://}"; pubhost="${pubhost%%[:/]*}"
priv_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [ -n "$priv_ip" ] && ! timeout 4 bash -c "</dev/tcp/$pubhost/22" 2>/dev/null; then
  log "检测到无法回环访问公网地址 $pubhost（云 NAT hairpin 受限），添加 iptables DNAT ..."
  for port in 8080 "$OIDC_PORT"; do
    if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
    $SUDO iptables -t nat -C OUTPUT -d "$pubhost" -p tcp --dport "$port" -j DNAT --to-destination "$priv_ip" 2>/dev/null || \
      $SUDO iptables -t nat -I OUTPUT 1 -d "$pubhost" -p tcp --dport "$port" -j DNAT --to-destination "$priv_ip"
    $SUDO iptables -t nat -C PREROUTING -d "$pubhost" -p tcp --dport "$port" -j DNAT --to-destination "$priv_ip" 2>/dev/null || \
      $SUDO iptables -t nat -I PREROUTING 1 -d "$pubhost" -p tcp --dport "$port" -j DNAT --to-destination "$priv_ip"
  done
  log "hairpin DNAT 完成（宿主机与容器均经内网 IP $priv_ip 回环；重启服务器后需重跑本脚本）"
fi

# ---------- 2. .env 生成/补全（不覆盖已有键） ----------
gen_secret() { openssl rand -hex 32 2>/dev/null || python3 -c 'import secrets; print(secrets.token_hex(32))'; }
touch .env
env_set() { grep -qE "^$1=" .env || printf '%s=%s\n' "$1" "$2" >> .env; }
env_warn_if_differs() {
  cur="$(grep -E "^$1=" .env | tail -1 | cut -d= -f2-)"
  if [ -n "$cur" ] && [ "$cur" != "$2" ]; then
    log "⚠️  .env 已有 $1=$cur，与派生值不同——保留原值（如需用测试 IdP 请手工改正）"
  fi
}
env_set AUTH_SECRET "$(gen_secret)"
env_set OIDC_PORT "$OIDC_PORT"
env_set REDIRECT_URI "$REDIRECT_URI";  env_warn_if_differs REDIRECT_URI "$REDIRECT_URI"
env_set OAUTH_SERVER_URL "$OAUTH_SERVER_URL"; env_warn_if_differs OAUTH_SERVER_URL "$OAUTH_SERVER_URL"
env_set CLIENT_ID "hvac-proxy"
env_set CLIENT_SECRET "hvac-proxy-secret"
env_set POSTGRES_PASSWORD "hvac_dev_password"
grep -qE '^AUTH_SECRET=.+' .env || die ".env 缺少 AUTH_SECRET"
log ".env 就绪"

# ---------- 3. 构建并启动 ----------
log "docker compose 构建启动（首次构建需拉取镜像与依赖，约 5~15 分钟）..."
docker compose -f docker-compose.yml -f docker-compose.oidc.yml up -d --build

# ---------- 4. 等待就绪 ----------
wait_url() { # url 名称 尝试次数
  local url="$1" name="$2" tries="${3:-80}" code
  for _ in $(seq 1 "$tries"); do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" 2>/dev/null || true)"
    [ "$code" = "200" ] && { log "$name 就绪"; return 0; }
    sleep 3
  done
  die "$name 未就绪（$url），请用 docker compose logs 排查"
}
wait_url "$PUB:$OIDC_PORT/.well-known/openid-configuration" "oidc (测试 IdP)"
wait_url "$PUB:8080/health" "proxy (网关)"
api_ready=0
for _ in $(seq 1 80); do
  if docker compose exec -T api python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)" 2>/dev/null; then
    api_ready=1; log "api 就绪"; break
  fi
  sleep 3
done
[ "$api_ready" = "1" ] || die "api 未就绪，请用 docker compose logs api 排查"

# ---------- 5. 冒烟验证 ----------
log "运行冒烟验证（无头登录 + 上传 + 确定性测算）..."
if bash scripts/verify_deployment.sh "$PUB:8080"; then
  log "🎉 部署完成：浏览器打开 $PUB:8080 登录（测试 IdP 账号 test / test123456）"
else
  die "冒烟验证未通过，请查看上方输出与 docker compose logs"
fi
