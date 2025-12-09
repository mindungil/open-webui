# /workspace/open-webui/sso/sso_server.py

from flask import Flask, request, redirect, make_response
import hashlib
import time

app = Flask(__name__)

# 포털 ↔ SSO 서버가 공유하는 시크릿
# (행정포탈 Java 코드의 SSO_SECRET 과 반드시 동일하게 맞출 것)
SSO_SECRET = "jb-portal-openwebui-2024-SSO-ONLY"

# SSO 토큰 유효 시간 (초) - 포털에서 만든 ts 기준
TOKEN_TTL = 300  # 5분

def create_token(uid, ts):
    """
    uid + "|" + ts + "|" + SSO_SECRET 를 SHA-1 해시해서 16진 문자열로 반환
    """
    data = "%s|%s|%s" % (uid, ts, SSO_SECRET)
    digest = hashlib.sha1(data.encode("utf-8")).hexdigest()
    return digest

def verify_sso(uid, ts, token):
    """
    포털에서 넘어온 uid/ts/token 이 유효한지 검증.
    """
    if not uid or not ts or not token:
        return False, "Invalid SSO request (missing params)"

    try:
        ts_val = int(ts)
    except ValueError:
        return False, "Invalid timestamp"

    now_ms = int(time.time() * 1000)
    # 토큰 생성 시각과 현재 시각 차이가 TOKEN_TTL 초를 넘으면 만료
    if abs(now_ms - ts_val) > TOKEN_TTL * 1000:
        return False, "SSO token expired"

    expected = create_token(uid, ts_val)
    if token != expected:
        return False, "SSO token mismatch"

    return True, ts_val

@app.route("/sso-login")
def sso_login():
    """
    행정포털에서 넘어오는 SSO 엔드포인트.
    - uid, ts, token 을 검증하고
    - 성공하면 ai_sso_user 쿠키를 심어주고 / 로 리다이렉트
    """

    uid   = request.args.get("uid", "")
    ts    = request.args.get("ts", "")
    token = request.args.get("token", "")

    ok, info = verify_sso(uid, ts, token)
    if not ok:
        # 검증 실패 시 400 에러와 이유 반환 (운영 시에는 로그만 찍고 일반 에러 페이지로 돌려도 됨)
        return info, 400

    # 여기까지 오면 포털이 uid 에 대해 인증했다는 것을 신뢰한다.
    # 이제 OpenWebUI 가 신뢰할 수 있는 SSO 쿠키를 발급해 준다.
    print("[SSO] SSO verified for uid=%s, ts=%s" % (uid, ts))

    # / 로 리다이렉트하면서 ai_sso_user 쿠키를 설정
    resp = make_response(redirect("/"))

    # ★ 여기 쿠키 이름/속성은 OpenWebUI 백엔드에서 읽어야 하므로,
    #    백엔드 코드에서 ai_sso_user 쿠키를 보고 로그인 처리하도록 구현 필요.
    resp.set_cookie(
        "ai_sso_user",
        uid,
        httponly=True,
        secure=True,
        samesite="Lax",
        path="/"
    )

    return resp

if __name__ == "__main__":
    # 개발/테스트용. 실제 운영 시에는 gunicorn/uWSGI 같은 WSGI 서버를 쓰는게 좋음.
    app.run(host="0.0.0.0", port=4000)

