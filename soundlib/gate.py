"""A simple shared password in front of the whole site (a stopgap until real accounts).

attach(api, password) makes every page and API call need a signed-in browser:
  - pages (/, /j/..., /edit/...) send signed-out visitors to the login page
  - API calls (/api/...) answer 401
  - POST /login  takes a normal form post with the field `password` (it may come from
    another site, such as a hosted login page). Right: sets the session cookie and
    goes to `/`. Wrong: goes back to the login page with ?error=1
  - GET /logout  signs out

The login page is the built-in web/login.html, or `login_url` when one is given (the
setting web_login_url). With no password set, the site is open (used for local testing).
The password itself is never in the code: on Modal it is the secret "site-password"
(key SITE_PASSWORD).
"""

import hashlib
import hmac
import time
from pathlib import Path

COOKIE = "asd_session"
KEEP_SECONDS = 30 * 24 * 3600


def session_token(password: str) -> str:
    return hmac.new(password.encode(), b"ai-sound-designer-session-v1", hashlib.sha256).hexdigest()


def attach(api, password: str, login_url: str = "", web_folder: Path | None = None) -> None:
    if not password:
        return
    from fastapi import Form, Request
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

    folder = web_folder or Path(__file__).resolve().parent.parent / "web"
    page = (folder / "login.html").read_text()
    token = session_token(password)
    login_at = login_url.strip() or "/login"

    def signed_in(request: Request) -> bool:
        return hmac.compare_digest(request.cookies.get(COOKIE, ""), token)

    @api.middleware("http")
    async def gate(request: Request, call_next):
        path = request.url.path
        if path in ("/login", "/logout") or signed_in(request):
            return await call_next(request)
        if path.startswith("/api/"):
            return JSONResponse({"detail": "Please sign in first."}, status_code=401)
        if login_url.strip():
            return RedirectResponse(login_at, status_code=302)
        return RedirectResponse(f"/login?next={path}" if path != "/" else "/login", status_code=302)

    @api.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        if signed_in(request):
            return RedirectResponse("/", status_code=302)
        return page

    @api.post("/login")
    def login(password_given: str = Form("", alias="password"), next: str = Form("/")):
        if not hmac.compare_digest(password_given.strip().encode(), password.encode()):
            time.sleep(1.0)   # slows down guessing
            return RedirectResponse(login_at + ("&" if "?" in login_at else "?") + "error=1", status_code=303)
        target = next if next.startswith("/") and not next.startswith("//") and "\\" not in next else "/"
        reply = RedirectResponse(target, status_code=303)
        reply.set_cookie(COOKIE, token, max_age=KEEP_SECONDS, httponly=True, secure=True, samesite="lax")
        return reply

    @api.get("/logout")
    def logout():
        reply = RedirectResponse(login_at, status_code=302)
        reply.delete_cookie(COOKIE)
        return reply
