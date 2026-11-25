"""Checks for the sign-in gate. Run: python -m pytest tests/ -q"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from soundlib import gate  # noqa: E402


def _site(**options):
    api = FastAPI()

    @api.get("/")
    def home():
        return {"ok": 1}

    @api.get("/api/x")
    def x():
        return {"ok": 1}

    gate.attach(api, "secret-word", **options)
    return TestClient(api, base_url="https://testserver", follow_redirects=False)


def test_shared_password():
    c = _site()
    assert c.get("/").status_code == 200 and c.get("/api/me").json()["signed_in"] is False
    assert c.get("/edit/abc").headers["location"] == "/login?next=/edit/abc"
    assert c.get("/api/x").status_code == 401 and c.get("/login").status_code == 200
    assert c.post("/login", data={"password": "nope"}).headers["location"] == "/login?error=1"
    assert c.post("/login", data={"password": "secret-word", "next": "//evil.com"}).headers["location"] == "/"
    assert c.get("/").status_code == 200 and c.get("/api/x").status_code == 200
    assert c.get("/logout").headers["location"] == "/"


def test_accounts():
    answers = {"good": ("yes", {"uid": "u1", "email": "a@b.c"}), "waiting": ("waitlist", {})}
    options = dict(login_url="https://signin.example", supabase_url="https://x.supabase.co",
                   supabase_anon_key="anon", check=lambda url, key, token: answers.get(token, ("error", {})))
    c = _site(**options)
    assert c.get("/").status_code == 200 and c.get("/edit/abc").headers["location"] == "https://signin.example"
    assert c.get("/api/me").json() == {"signed_in": False, "email": "", "login_url": "https://signin.example", "open": False}
    assert c.get("/api/x").status_code == 401
    assert c.post("/login", data={"password": "secret-word"}).headers["location"] == "https://signin.example?error=1"
    assert c.post("/login", data={"token": "waiting"}).headers["location"] == "https://signin.example?waitlist=1"
    assert c.post("/login", data={"token": "bad"}).headers["location"] == "https://signin.example?error=1"
    assert c.get("/api/x").status_code == 401
    assert c.post("/login", data={"token": "good"}).headers["location"] == "/"
    assert c.get("/api/me").json()["email"] == "a@b.c" and c.get("/api/x").status_code == 200
    forged = _site(**options)
    forged.cookies.set(gate.COOKIE, gate.session_token("secret-word"))   # the old shared cookie
    assert forged.get("/api/x").status_code == 401
    assert c.get("/logout").headers["location"] == "/"
