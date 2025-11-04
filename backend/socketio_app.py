import socketio
from open_webui.main import app as fastapi_app
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*",
                           ping_interval=25, ping_timeout=60)
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app, socketio_path="socket.io")
