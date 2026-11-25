"""Sign in to Google Cloud from a cloud session, in two short steps that survive the
session restarting in between (plain `gcloud auth login` has to stay running).

  python scripts/google_login.py start         prints a link for Anthony to open
  python scripts/google_login.py finish CODE   CODE = what Google shows him afterwards

The sign-in is saved where Google's tools look for it
(~/.config/gcloud/application_default_credentials.json). Nothing goes into git.
`python scripts/google_login.py token` prints a fresh access token for REST calls.
"""
import base64, hashlib, json, os, secrets, sys, urllib.parse, urllib.request

# The Google Cloud command-line tool's own public sign-in identity (from its source).
CLIENT_ID = "32555940559.apps.googleusercontent.com"
CLIENT_SECRET = "ZmssLNjJy2998hD4CTg2ejr2"
REDIRECT = "https://sdk.cloud.google.com/authcode.html"
SCOPES = "openid https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/cloud-platform"
STATE = os.path.join(os.environ.get("TMPDIR", "/tmp"), "google_login_flow.json")
ADC = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")


def _post(data: dict) -> dict:
    req = urllib.request.Request("https://oauth2.googleapis.com/token",
                                 data=urllib.parse.urlencode(data).encode())
    try:
        return json.load(urllib.request.urlopen(req, timeout=30))
    except urllib.error.HTTPError as e:
        sys.exit(f"Google refused: {e.read().decode()[:300]}")


def start() -> None:
    verifier = secrets.token_urlsafe(64)
    state = secrets.token_urlsafe(16)
    with open(STATE, "w") as f:
        json.dump({"verifier": verifier, "state": state}, f)
    os.chmod(STATE, 0o600)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    print("https://accounts.google.com/o/oauth2/auth?" + urllib.parse.urlencode({
        "response_type": "code", "client_id": CLIENT_ID, "redirect_uri": REDIRECT, "scope": SCOPES,
        "state": state, "access_type": "offline", "prompt": "consent",
        "code_challenge": challenge, "code_challenge_method": "S256"}))


def finish(code: str) -> None:
    flow = json.load(open(STATE))
    tok = _post({"grant_type": "authorization_code", "code": code.strip(), "client_id": CLIENT_ID,
                 "client_secret": CLIENT_SECRET, "redirect_uri": REDIRECT,
                 "code_verifier": flow["verifier"]})
    os.makedirs(os.path.dirname(ADC), exist_ok=True)
    with open(ADC, "w") as f:
        json.dump({"type": "authorized_user", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
                   "refresh_token": tok["refresh_token"]}, f)
    os.chmod(ADC, 0o600)
    os.remove(STATE)
    print("Signed in to Google Cloud.")


def token() -> str:
    adc = json.load(open(ADC))
    return _post({"grant_type": "refresh_token", "refresh_token": adc["refresh_token"],
                  "client_id": adc["client_id"], "client_secret": adc["client_secret"]})["access_token"]


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "start":
        start()
    elif what == "finish" and len(sys.argv) > 2:
        finish(sys.argv[2])
    elif what == "token":
        print(token())
    else:
        sys.exit(__doc__)
