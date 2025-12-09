#!/usr/bin/env bash
# /workspace/open-webui/backend/restart_backend.sh
set -euo pipefail

### ── 설정 ─────────────────────────────────────────────────────────────
ROOT_DIR="/workspace/open-webui/backend"
VENV_DIR="$ROOT_DIR/venv"
APP_IMPORT="socketio_app:app"     # dev.sh와 동일
PORT="${PORT:-8080}"              # 환경변수 PORT가 없으면 8080
RELOAD="${RELOAD:-1}"             # 1이면 --reload 사용, 0이면 미사용
HOST="0.0.0.0"
FORWARDED_ALLOW_IPS="*"

# dev.sh에서 쓰던 환경변수들 (없으면 기본값 채워줌)
export CORS_ALLOW_ORIGIN="${CORS_ALLOW_ORIGIN:-http://ai.jb.go.kr,https://ai.jb.go.kr,http://localhost:80,http://localhost:8080}"
export CORS_ALLOW_METHODS="${CORS_ALLOW_METHODS:-GET,POST,PUT,DELETE,OPTIONS}"
export CORS_ALLOW_HEADERS="${CORS_ALLOW_HEADERS:-*}"
export DATABASE_URL="${DATABASE_URL:-postgresql://admin:wjsqnrai@172.17.0.1:5432/webui}"

# 선택: 프론트 기준 URL(필요시 켜세요)
export WEBUI_URL="${WEBUI_URL:-https://ai.jb.go.kr}"

# 선택: .env 존재 시 불러오기 (키=값 형식)
ENV_FILE="$ROOT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  echo "[i] Loading .env from $ENV_FILE"
  # shellcheck disable=SC2046
  export $(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE" | xargs) || true
fi

### ── 함수 ─────────────────────────────────────────────────────────────
log() { printf "\033[1;34m[restart]\033[0m %s\n" "$*"; }

stop_backend() {
  log "Stopping old backend processes (TERM → KILL)..."
  pkill -TERM -f 'uvicorn|socketio_app|open_webui' || true
  sleep 3
  if pgrep -fa 'uvicorn|socketio_app|open_webui' >/dev/null 2>&1; then
    log "Forcing kill..."
    pkill -KILL -f 'uvicorn|socketio_app|open_webui' || true
  fi

  if ss -ltnp 2>/dev/null | grep -q ":$PORT"; then
    log "Port :$PORT still in use. Showing holders:"
    ss -ltnp | grep ":$PORT" || true
    exit 1
  fi
  log "OK - port :$PORT is free."
}

start_backend() {
  log "Activating venv..."
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  which uvicorn >/dev/null || { echo "uvicorn not found in venv"; exit 1; }

  log "Starting backend on :$PORT (reload=${RELOAD})..."
  CMD=( uvicorn "$APP_IMPORT" --port "$PORT" --host "$HOST" --forwarded-allow-ips "$FORWARDED_ALLOW_IPS" )
  [[ "$RELOAD" == "1" ]] && CMD+=( --reload )

  # 포그라운드 실행(로그 확인 원하면 그대로), 백그라운드 원하면 아래 exec 대신 nohup로 교체
  exec "${CMD[@]}"
}

health_check() {
  # 포그라운드 모드에선 이 함수가 도달하지 않지만, 백그라운드로 바꿀 경우 사용 가능
  log "Waiting for health endpoint..."
  for _ in {1..30}; do
    if curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null; then
      log "OK - http://127.0.0.1:${PORT}/api/health"
      break
    fi
    sleep 1
  done
}

### ── 본 실행 ─────────────────────────────────────────────────────────
cd "$ROOT_DIR"
log "Working directory: $(pwd)"
stop_backend
# start_backend가 exec로 프로세스를 대체하여 이후 줄은 실행되지 않습니다.
start_backend

# (참고) 백그라운드로 돌리려면 위의 exec를 다음으로 바꾸세요:
# nohup "${CMD[@]}" >/tmp/openwebui-backend.log 2>&1 &
# health_check
# log "Tail: /tmp/openwebui-backend.log"
# tail -n +1 -f /tmp/openwebui-backend.log


# SSO Server
python /workspace/open-webui/sso/sso_server.py &

