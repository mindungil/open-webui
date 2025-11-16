from open_webui.socket.main import sio
from open_webui.main import app as fastapi_app
import socketio

app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app, socketio_path="/ws/socket.io")
