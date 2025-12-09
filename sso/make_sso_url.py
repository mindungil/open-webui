import hashlib
import time
import urllib.parse

SSO_SECRET = "jb-portal-openwebui-2024-SSO-ONLY"  # 서버에 넣은 값과 반드시 동일
BASE_URL = "https://ai.jb.go.kr/sso-login"

def create_token(uid, ts):
    data = "%s|%s|%s" % (uid, ts, SSO_SECRET)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()

if __name__ == "__main__":
    uid = "admin@korea.kr"   # 여기만 임시로 원하는 아이디로
    ts  = int(time.time() * 1000)

    token = create_token(uid, ts)

    params = {
        "uid": uid,
        "ts": str(ts),
        "token": token
    }

    query = urllib.parse.urlencode(params)
    url = BASE_URL + "?" + query

    print("Generated SSO URL:")
    print(url)
