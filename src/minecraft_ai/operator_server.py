# ruff: noqa: E501
from __future__ import annotations

import ipaddress
import json
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from .agent_lifecycle import AgentProcess, agent_alive, stop_agent_process
from .config import app_paths
from .emergency import emergency_reason, emergency_stop_latched
from .platforms import discover_bedrock_linux_install, find_bedrock_linux_instances
from .social import OperatorMessage, OperatorMessageKind
from .storage import StateDatabase
from .supervisor import STATUS_FILE, send_command, supervisor_alive
from .telemetry import read_telemetry


MAX_BODY_BYTES = 16 * 1024


def _read_json_file(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _agent_status() -> dict[str, object]:
    try:
        process = AgentProcess.load()
    except (OSError, ValueError, TypeError, KeyError):
        return {"alive": False}
    return {
        "alive": agent_alive(process),
        "pid": process.pid,
        "role": process.role,
        "display": process.display,
        "window_id": process.window_id,
    }


def operator_status() -> dict[str, object]:
    reachable = supervisor_alive()
    if reachable:
        try:
            supervisor = send_command("status")
        except Exception:
            reachable = False
            supervisor = _read_json_file(STATUS_FILE) or {"state": "UNKNOWN"}
    else:
        supervisor = _read_json_file(STATUS_FILE) or {"state": "STOPPED"}
    install = discover_bedrock_linux_install()
    build = None if install is None else install.selected_build
    return {
        "server_time_ns": time.time_ns(),
        "supervisor": supervisor,
        "supervisor_reachable": reachable,
        "agent": _agent_status(),
        "telemetry": read_telemetry(),
        "bedrock": {
            "launcher": None if install is None else install.launcher_command,
            "version": None if build is None else build.version,
            "instances": [instance.instance_id for instance in find_bedrock_linux_instances()],
        },
        "emergency_stop": {
            "latched": emergency_stop_latched(),
            "reason": emergency_reason(),
        },
    }


class OperatorRequestHandler(BaseHTTPRequestHandler):
    server_version = "MinecraftAIOperator/0.1"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(HTTPStatus.OK, DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send_json(HTTPStatus.OK, operator_status())
        elif path == "/api/messages":
            with StateDatabase(app_paths().state_db) as database:
                messages = database.load_operator_messages(limit=100)
            self._send_json(
                HTTPStatus.OK,
                {"messages": [message.model_dump(mode="json") for message in messages]},
            )
        elif path == "/healthz":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._read_body()
            if path == "/api/messages":
                self._post_message(payload)
            elif path == "/api/control/pause":
                self._pause_agent()
            elif path == "/api/control/resume":
                self._resume_supervisor()
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ValueError, TypeError, KeyError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": f"{type(exc).__name__}: {exc}"},
            )

    def _read_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "")
        if not raw_length.isdigit():
            raise ValueError("Content-Length is required")
        length = int(raw_length)
        if length < 2 or length > MAX_BODY_BYTES:
            raise ValueError("request body size is invalid")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise TypeError("request body must be a JSON object")
        return cast(dict[str, Any], payload)

    def _post_message(self, payload: dict[str, Any]) -> None:
        allowed = {"text", "kind", "priority", "author"}
        unexpected = sorted(set(payload) - allowed)
        if unexpected:
            raise ValueError(f"unsupported fields: {', '.join(unexpected)}")
        message = OperatorMessage(
            message_id=uuid.uuid4().hex,
            created_ns=time.time_ns(),
            text=str(payload.get("text", "")).strip(),
            kind=OperatorMessageKind(str(payload.get("kind", "instruction"))),
            priority=float(payload.get("priority", 0.8)),
            author=str(payload.get("author", "operator")).strip()[:80] or "operator",
        )
        with StateDatabase(app_paths().state_db) as database:
            database.save_operator_message(message)
        self._send_json(HTTPStatus.CREATED, message.model_dump(mode="json"))

    def _pause_agent(self) -> None:
        stop_agent_process()
        result: dict[str, object]
        if supervisor_alive():
            result = send_command("pause")
        else:
            result = {"state": "STOPPED"}
        self._send_json(HTTPStatus.OK, result)

    def _resume_supervisor(self) -> None:
        if emergency_stop_latched():
            raise ValueError("emergency stop is latched")
        if not supervisor_alive():
            raise ValueError("supervisor is not running; use the CLI to start it")
        self._send_json(HTTPStatus.OK, send_command("resume"))

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        self._send_bytes(
            status,
            json.dumps(payload, separators=(",", ":")).encode(),
            "application/json; charset=utf-8",
        )

    def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def serve_operator_dashboard(*, host: str = "127.0.0.1", port: int = 8765) -> None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("dashboard host must be a numeric loopback address") from exc
    if not address.is_loopback:
        raise ValueError("dashboard is local-only and must bind to a loopback address")
    if not 1 <= port <= 65535:
        raise ValueError("dashboard port must be between 1 and 65535")
    server = ThreadingHTTPServer((host, port), OperatorRequestHandler)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Minecraft AI Operator</title>
<style>
:root{color-scheme:dark;--bg:#07100d;--panel:#101c17;--line:#263b32;--text:#ecf8f1;
--muted:#95aa9f;--green:#5ce58c;--amber:#ffc766;--red:#ff6b72;--cyan:#70d7e8}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% -10%,#18392a,transparent 32%),
var(--bg);font:15px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--text)}
main{max-width:1400px;margin:auto;padding:28px}.top{display:flex;align-items:center;gap:16px;margin-bottom:24px}
.mark{width:48px;height:48px;background:linear-gradient(145deg,#75ef9e,#16844a);border-radius:13px;
box-shadow:0 0 35px #3cd47745;display:grid;place-items:center;font-size:25px}.title h1{font-size:20px;margin:0}
.title p{margin:4px 0 0;color:var(--muted)}.live{margin-left:auto;padding:7px 11px;border:1px solid var(--line);
border-radius:999px;color:var(--muted)}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;
background:var(--red);margin-right:8px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:13px}
.card{background:#101c17dd;border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 16px 50px #0003}
.label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.11em}.value{font-size:22px;margin-top:8px}
.wide{grid-column:span 2}.full{grid-column:1/-1}.two{display:grid;grid-template-columns:1.05fr .95fr;gap:13px;margin-top:13px}
h2{font-size:14px;margin:0 0 14px}textarea,input,select{width:100%;background:#07110d;border:1px solid var(--line);
color:var(--text);border-radius:9px;padding:11px;font:inherit}textarea{min-height:112px;resize:vertical}.row{display:flex;gap:9px;margin-top:9px}
.row>*{flex:1}button{border:1px solid #3da867;background:#1e7a45;color:white;padding:10px 14px;border-radius:9px;
font:inherit;font-weight:700;cursor:pointer}button.secondary{background:#17251f;border-color:var(--line)}button.warn{background:#5e471d;border-color:#9f7c35}
button:hover{filter:brightness(1.13)}.feed{max-height:330px;overflow:auto}.msg{border-top:1px solid var(--line);padding:12px 0}
.msg:first-child{border:0}.meta{display:flex;gap:8px;color:var(--muted);font-size:11px}.kind{color:var(--cyan)}.text{margin-top:6px;white-space:pre-wrap;overflow-wrap:anywhere}
.reason{font-size:16px;margin-top:9px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:13px}
.metric{background:#09130f;border:1px solid #1d3028;border-radius:8px;padding:9px}.metric b{display:block;font-size:17px;margin-top:3px}
.ok{color:var(--green)}.bad{color:var(--red)}.amber{color:var(--amber)}#notice{min-height:20px;margin-top:9px;color:var(--muted)}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}.wide{grid-column:span 2}}
@media(max-width:540px){main{padding:16px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:1}.row{flex-direction:column}.live{display:none}}
</style></head>
<body><main><div class="top"><div class="mark">⛏</div><div class="title"><h1>Minecraft AI Operator</h1>
<p>Live state, cognition, and durable operator input</p></div><div class="live"><span class="dot" id="dot"></span><span id="connection">Connecting</span></div></div>
<section class="grid"><div class="card"><div class="label">Bedrock</div><div class="value" id="bedrock">—</div><div id="version" class="label"></div></div>
<div class="card"><div class="label">Supervisor</div><div class="value" id="supervisor">—</div></div>
<div class="card"><div class="label">Agent</div><div class="value" id="agent">—</div></div>
<div class="card"><div class="label">Active skill</div><div class="value" id="skill">—</div></div>
<div class="card wide"><div class="label">Current goal</div><div class="value" id="goal">No active goal</div></div>
<div class="card wide"><div class="label">Latest cognition</div><div class="reason" id="reason">Waiting for agent telemetry.</div></div></section>
<section class="two"><div class="card"><h2>Talk to the high-level agent</h2><textarea id="text" maxlength="2000" placeholder="Give an instruction, ask a question, provide feedback, or correct its current approach…"></textarea>
<div class="row"><select id="kind"><option value="instruction">Instruction</option><option value="question">Question</option><option value="feedback">Feedback</option><option value="correction">Correction</option></select>
<select id="priority"><option value="0.8">High priority</option><option value="0.55">Normal priority</option><option value="1">Urgent</option></select><button id="send">Send to agent</button></div>
<div id="notice"></div><div class="row"><button class="warn" id="pause">Pause & revoke control</button><button class="secondary" id="resume">Resume supervisor safely</button></div>
<div class="metrics"><div class="metric"><span class="label">Frames</span><b id="frames">0</b></div><div class="metric"><span class="label">Actions</span><b id="actions">0</b></div><div class="metric"><span class="label">Capture</span><b id="capture">—</b></div></div></div>
<div class="card"><h2>Operator conversation</h2><div class="feed" id="feed"><div class="label">No messages yet</div></div></div></section></main>
<script>
const $=id=>document.getElementById(id);const esc=v=>v??'—';
async function api(path,options){const r=await fetch(path,options);const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d}
function cls(el,good){el.className='value '+(good?'ok':'bad')}
async function refresh(){try{const s=await api('/api/status');$('dot').style.background='var(--green)';$('connection').textContent='Live telemetry';
const bi=s.bedrock.instances.length>0; $('bedrock').textContent=bi?'RUNNING':'STOPPED';cls($('bedrock'),bi);$('version').textContent=esc(s.bedrock.version);
const sup=esc(s.supervisor.state);$('supervisor').textContent=sup;cls($('supervisor'),s.supervisor_reachable&&sup!=='FAILSAFE');
const alive=s.agent.alive;$('agent').textContent=alive?'RUNNING':'DISARMED';cls($('agent'),alive);const t=s.telemetry||{};
$('skill').textContent=esc(t.active_skill);$('goal').textContent=esc(t.chosen_goal_id||'No active goal');$('reason').textContent=esc(t.reasoning_summary||'Waiting for agent telemetry.');
$('frames').textContent=esc(t.frames||0);$('actions').textContent=esc(t.motor_actions||0);$('capture').textContent=t.last_capture_ms==null?'—':t.last_capture_ms+' ms';
}catch(e){$('dot').style.background='var(--red)';$('connection').textContent='Disconnected'}}
async function messages(){try{const d=await api('/api/messages');const feed=$('feed');feed.replaceChildren();if(!d.messages.length){const x=document.createElement('div');x.className='label';x.textContent='No messages yet';feed.append(x);return}
d.messages.forEach(m=>{const box=document.createElement('div');box.className='msg';const meta=document.createElement('div');meta.className='meta';const k=document.createElement('span');k.className='kind';k.textContent=m.kind;const when=document.createElement('span');when.textContent=new Date(m.created_ns/1e6).toLocaleTimeString();const status=document.createElement('span');status.textContent=m.status;meta.append(k,when,status);const text=document.createElement('div');text.className='text';text.textContent=m.text;box.append(meta,text);if(m.response_text){const reply=document.createElement('div');reply.className='text ok';reply.textContent='Agent: '+m.response_text;box.append(reply)}feed.append(box)})}catch(e){}}
$('send').onclick=async()=>{const text=$('text').value.trim();if(!text)return;try{await api('/api/messages',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,kind:$('kind').value,priority:Number($('priority').value)})});$('text').value='';$('notice').textContent='Delivered to the durable cognition inbox.';messages()}catch(e){$('notice').textContent=e.message}};
$('pause').onclick=async()=>{try{await api('/api/control/pause',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});$('notice').textContent='Agent paused; motor capability revoked.';refresh()}catch(e){$('notice').textContent=e.message}};
$('resume').onclick=async()=>{try{await api('/api/control/resume',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});$('notice').textContent='Supervisor returned to safe idle. Live control was not armed.';refresh()}catch(e){$('notice').textContent=e.message}};
refresh();messages();setInterval(refresh,500);setInterval(messages,2000);
</script></body></html>"""
