"""Sign in to Modal from a cloud session: prints a link for Anthony to approve, then
waits with short polls that survive dropped connections. Run it in the background.
Usage: python scripts/modal_login.py"""
import asyncio, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soundlib import proxyfix
proxyfix.apply()
from modal.client import _Client
from modal.config import config
from modal_proto import api_pb2

S = os.environ.get("TMPDIR", "/tmp")
STATE = f"{S}/modal_login_flow.json"

async def main():
    url = config.get("server_url")
    if not os.path.exists(STATE):
        async with _Client.anonymous(url) as c:
            r = await c._stub.TokenFlowCreate(api_pb2.TokenFlowCreateRequest(localhost_port=0))
            json.dump({"id": r.token_flow_id, "secret": r.wait_secret, "url": r.web_url, "code": r.code}, open(STATE, "w"))
            os.chmod(STATE, 0o600)
    st = json.load(open(STATE))
    print("URL", st["url"], "CODE", st["code"], flush=True)
    deadline = time.time() + 6600
    while time.time() < deadline:
        try:
            async with _Client.anonymous(url) as c:
                while time.time() < deadline:
                    req = api_pb2.TokenFlowWaitRequest(token_flow_id=st["id"], timeout=8.0, wait_secret=st["secret"])
                    resp = await c._stub.TokenFlowWait(req, retry=None, timeout=14.0)
                    if not resp.timeout:
                        from modal.token_flow import _set_token
                        await _set_token(resp.token_id, resp.token_secret, profile=None, activate=True, verify=False, server_url=url)
                        os.remove(STATE)
                        print("DONE workspace", resp.workspace_username, flush=True)
                        return 0
        except Exception as e:
            print("retry after", type(e).__name__, str(e)[:100], flush=True)
            await asyncio.sleep(2)
    print("GAVE UP", flush=True)
    return 1

sys.exit(asyncio.run(main()))
