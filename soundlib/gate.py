"""Who may use the site. Two modes, chosen by the settings:

1. One shared password (the stopgap). On when the Modal secret "site-password"
   (key SITE_PASSWORD) has a value and no accounts are set up.
     POST /login  form field `password`.

2. Personal accounts (Supabase, with the sign-in page hosted elsewhere, see
   web_login_url). On when web_supabase_url and web_supabase_anon_key are set.
     POST /login  form field `token`: the person's Supabase access token, sent by a
     full-page form post from the sign-in page. The server asks Supabase whether this
     person may use the app (the database function can_use_app). Yes: signed in.
     No: back to the sign-in page with ?waitlist=1. Anything else: ?error=1.
   The shared password is no longer accepted in this mode.

In both modes:
  - the front page is open to everyone (PUBLIC below); it shows a "Sign in" button and
    asks for sign-in when someone submits a video
  - other pages (/j/..., /edit/...) send signed-out visitors to the sign-in page
  - other API calls (/api/..., so also the upload) answer 401
  - GET /api/me tells the page who is signed in and where to sign in
  - a right answer sets a signed, HttpOnly session cookie and goes to `/`
  - GET /logout signs out and goes back to the front page

The sign-in page is the built-in web/login.html, or web_login_url when set. With
neither a password nor accounts, the site is open (used for local testing). Cookies are
signed with a key made from SITE_PASSWORD, so that secret must exist in both modes.
"""

import base64
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

COOKIE = "asd_session"
PASSWORD_KEEP_SECONDS = 30 * 24 * 3600
ACCOUNT_KEEP_SECONDS = 7 * 24 * 3600     # short, so taking away someone's access bites soon
PUBLIC = ("/", "/api/config", "/api/me", "/favicon.ico")   # what signed-out visitors may load


def session_token(password: str) -> str:
    return hmac.new(password.encode(), b"ai-sound-designer-session-v1", hashlib.sha256).hexdigest()


def _sign(key: bytes, payload: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    return body + "." + hmac.new(key, body.encode(), hashlib.sha256).hexdigest()


def _read(key: bytes, cookie: str) -> dict | None:
    """The payload of a signed cookie, or None if it is forged, broken or expired."""
    body, _, mac = cookie.partition(".")
    if not mac or not hmac.compare_digest(mac, hmac.new(key, body.encode(), hashlib.sha256).hexdigest()):
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except ValueError:
        return None
    return payload if isinstance(payload, dict) and payload.get("exp", 0) > time.time() else None


def ask_supabase(supabase_url: str, anon_key: str, token: str) -> tuple[str, dict]:
    """Is this person allowed in? Returns ("yes" | "waitlist" | "error", who they are)."""
    base = supabase_url.rstrip("/")
    headers = {"apikey": anon_key, "Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def call(path: str, data: bytes | None):
        request = urllib.request.Request(base + path, data=data, headers=headers,
                                         method="POST" if data is not None else "GET")
        with urllib.request.urlopen(request, timeout=8) as reply:
            return json.loads(reply.read().decode() or "null")

    try:
        allowed = call("/rest/v1/rpc/can_use_app", b"{}")
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return "error", {}
    if allowed is not True:
        return ("waitlist" if allowed is False else "error"), {}
    try:
        user = call("/auth/v1/user", None) or {}
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        user = {}
    return "yes", {"uid": str(user.get("id", ""))[:64], "email": str(user.get("email", ""))[:200]}


def attach(api, password: str, login_url: str = "", web_folder: Path | None = None,
           supabase_url: str = "", supabase_anon_key: str = "", check=ask_supabase) -> None:
    accounts = bool(supabase_url.strip() and supabase_anon_key.strip())
    if not password:
        if accounts:
            raise RuntimeError("Accounts need the SITE_PASSWORD secret too: it signs the cookies.")

        @api.get("/api/me")
        def me_open():
            return {"signed_in": True, "email": "", "login_url": "", "open": True}
        return
    from fastapi import Form, Request
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

    folder = web_folder or Path(__file__).resolve().parent.parent / "web"
    page = (folder / "login.html").read_text()
    shared = session_token(password)
    key = hashlib.sha256(b"ai-sound-designer-cookie-key:" + password.encode()).digest()
    external = login_url.strip()
    login_at = external or "/login"
    back = lambda why: RedirectResponse(login_at + ("&" if "?" in login_at else "?") + why + "=1", status_code=303)

    def who(request: Request) -> dict | None:
        """The signed-in person ({} in shared-password mode), or None."""
        cookie = request.cookies.get(COOKIE, "")
        if accounts:
            return _read(key, cookie)
        return {} if hmac.compare_digest(cookie, shared) else None

    @api.middleware("http")
    async def gate(request: Request, call_next):
        path = request.url.path
        person = who(request)
        if path in ("/login", "/logout") or path in PUBLIC or person is not None:
            request.state.person = person or {}
            return await call_next(request)
        if path.startswith("/api/"):
            return JSONResponse({"detail": "Please sign in first."}, status_code=401)
        if external:
            return RedirectResponse(external, status_code=302)
        return RedirectResponse(f"/login?next={path}" if path != "/" else "/login", status_code=302)

    @api.get("/api/me")
    def me(request: Request):
        person = who(request)
        reply = JSONResponse({"signed_in": person is not None, "email": (person or {}).get("email", ""),
                              "login_url": login_at, "open": False})
        reply.headers["Cache-Control"] = "no-store"
        return reply

    @api.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        if who(request) is not None:
            return RedirectResponse("/", status_code=302)
        if external:
            return RedirectResponse(external, status_code=302)
        return page

    @api.post("/login")
    def login(password_given: str = Form("", alias="password"), token: str = Form(""),
              next: str = Form("/")):
        if accounts:
            answer, person = check(supabase_url, supabase_anon_key, token.strip()) if token.strip() else ("error", {})
            if answer != "yes":
                return back("waitlist" if answer == "waitlist" else "error")
            value, keep = _sign(key, {**person, "exp": int(time.time()) + ACCOUNT_KEEP_SECONDS}), ACCOUNT_KEEP_SECONDS
        else:
            if not hmac.compare_digest(password_given.strip().encode(), password.encode()):
                time.sleep(1.0)   # slows down guessing
                return back("error")
            value, keep = shared, PASSWORD_KEEP_SECONDS
        target = next if next.startswith("/") and not next.startswith("//") and "\\" not in next else "/"
        reply = RedirectResponse(target, status_code=303)
        reply.set_cookie(COOKIE, value, max_age=keep, httponly=True, secure=True, samesite="lax")
        return reply

    @api.get("/logout")
    def logout():
        reply = RedirectResponse("/", status_code=302)
        reply.delete_cookie(COOKIE)
        return reply
