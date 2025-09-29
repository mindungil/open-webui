export CORS_ALLOW_ORIGIN="http://localhost:5173;http://localhost:8080;http://220.124.155.35:5173;http://220.124.155.35/8081"
PORT="${PORT:-8080}"
uvicorn open_webui.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips '*' --reload
