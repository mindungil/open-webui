export CORS_ALLOW_ORIGIN="http://ai.jb.go.kr;https://ai.jb.go.kr;http://localhost:80;http://localhost:8080"
export CORS_ALLOW_METHODS="GET,POST,PUT,DELETE,OPTIONS"
export CORS_ALLOW_HEADERS="*"
export DATABASE_URL="postgresql://admin:wjsqnrai@172.17.0.1:5432/webui"


PORT="${PORT:-8080}"
uvicorn open_webui.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips '*' --reload
