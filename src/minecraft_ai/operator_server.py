# ruff: noqa: E501
from __future__ import annotations

import ipaddress
import json
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from PIL import Image

from .agent_lifecycle import AgentProcess, agent_alive, stop_agent_process
from .config import app_paths
from .emergency import emergency_reason, emergency_stop_latched
from .perception import ScreenRegion, Track
from .perception_service import frame_dhash
from .platforms import discover_bedrock_linux_install, find_bedrock_linux_instances
from .platforms.bedrock_x11 import IsolatedX11Capture
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


def _current_agent_frame_dhash() -> str | None:
    """Bind an operator screen region to the live view it was selected from.

    The target can still be stored when no capture process is available (for
    offline UI/state inspection), but only a live target receives the visual
    reference needed for durable grounded-policy admission.
    """
    try:
        process = AgentProcess.load()
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if not agent_alive(process):
        return None
    capture = IsolatedX11Capture(
        process.display,
        process.window_id,
        allow_host=False,
    )
    try:
        return frame_dhash(capture.capture())
    except (OSError, RuntimeError, ValueError):
        return None
    finally:
        capture.close()


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
        elif path == "/api/frame.png":
            self._get_frame()
        elif path == "/api/target":
            with StateDatabase(app_paths().state_db) as database:
                target = database.load_operator_target()
            self._send_json(
                HTTPStatus.OK,
                {"target": None if target is None else target.model_dump(mode="json")},
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
            elif path == "/api/target":
                self._post_target(payload)
            elif path == "/api/target/clear":
                self._clear_target()
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

    def _get_frame(self) -> None:
        try:
            process = AgentProcess.load()
        except (OSError, ValueError, TypeError, KeyError):
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "agent capture process is not running"},
            )
            return
        if not agent_alive(process):
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "agent capture process is not running"},
            )
            return
        capture = IsolatedX11Capture(
            process.display,
            process.window_id,
            allow_host=False,
        )
        try:
            frame = capture.capture()
        finally:
            capture.close()
        image = Image.frombytes(
            "RGBA",
            (frame.width, frame.height),
            frame.bgra,
            "raw",
            "BGRA",
        ).convert("RGB")
        output = BytesIO()
        image.save(output, format="JPEG", quality=86, optimize=True)
        self._send_bytes(HTTPStatus.OK, output.getvalue(), "image/jpeg")

    def _post_target(self, payload: dict[str, Any]) -> None:
        allowed = {"label", "x", "y", "width", "height"}
        unexpected = sorted(set(payload) - allowed)
        if unexpected:
            raise ValueError(f"unsupported fields: {', '.join(unexpected)}")
        label = str(payload.get("label", "target")).strip()[:80]
        if not label:
            raise ValueError("target label is required")
        now_ns = time.monotonic_ns()
        attributes: dict[str, str | int | float | bool] = {
            "source": "operator",
            "grounding": "explicit-region",
        }
        reference_dhash = _current_agent_frame_dhash()
        if reference_dhash is not None:
            attributes["reference_dhash"] = reference_dhash
        target = Track(
            track_id=f"operator:{uuid.uuid4().hex}",
            label=label,
            confidence=1.0,
            region=ScreenRegion(
                x=float(payload["x"]),
                y=float(payload["y"]),
                width=float(payload["width"]),
                height=float(payload["height"]),
            ),
            first_seen_ns=now_ns,
            last_seen_ns=now_ns,
            attributes=attributes,
        )
        with StateDatabase(app_paths().state_db) as database:
            database.save_operator_target(target)
        self._send_json(HTTPStatus.CREATED, target.model_dump(mode="json"))

    def _clear_target(self) -> None:
        with StateDatabase(app_paths().state_db) as database:
            database.clear_operator_target()
        self._send_json(HTTPStatus.OK, {"target": None})

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
.facts{margin:9px 0 0;max-height:220px;overflow:auto;white-space:pre-wrap;color:#b8d8ca;font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}
.metric{background:#09130f;border:1px solid #1d3028;border-radius:8px;padding:9px}.metric b{display:block;font-size:17px;margin-top:3px}
.viewer{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:10px;background:#030705;line-height:0;user-select:none;touch-action:none}
.viewer img{display:block;width:100%;height:auto;min-height:220px;object-fit:contain}.selection{position:absolute;border:2px solid var(--cyan);background:#70d7e829;box-shadow:0 0 0 1px #001a;display:none;pointer-events:none}
.prediction{position:absolute;border:2px dashed var(--amber);background:#ffc76618;box-shadow:0 0 12px #ffc76688;display:none;pointer-events:none}
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
<div class="card wide"><div class="label">Learned motor instruction</div><div class="reason" id="instruction">Waiting for an executable option.</div><div class="label" id="constraints"></div></div>
<div class="card"><div class="label">Motor policy</div><div class="value" id="policy">—</div><div id="policyMetrics" class="label"></div></div>
<div class="card"><div class="label">Camera state</div><div class="value" id="camera">—</div><div id="cameraMode" class="label"></div></div>
<div class="card wide"><div class="label">Latest cognition</div><div class="reason" id="reason">Waiting for agent telemetry.</div><div class="label" id="outcome"></div></div></section>
<section class="card"><h2>Ground ROCKET-2 on the live Bedrock frame</h2><div class="viewer" id="viewer"><img id="worldFrame" alt="Live Bedrock frame" draggable="false"><div class="selection" id="selection"></div><div class="prediction" id="prediction" title="ROCKET-2 learned current-view prediction"></div></div>
<div class="row"><input id="targetLabel" maxlength="80" value="log" aria-label="Target label"><button id="setTarget">Run ROCKET-2 on selection</button><button class="secondary" id="clearTarget">Clear target</button></div><div id="targetNotice" class="label">Drag a tight box around the object, then submit it as ROCKET-2's cross-view reference.</div></section>
<section class="card"><h2>Fresh learned perception</h2><pre class="facts" id="facts">Waiting for perception facts.</pre></section>
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
let dragging=false,startX=0,startY=0,targetBox=null;const viewer=$('viewer'),selection=$('selection'),prediction=$('prediction'),worldFrame=$('worldFrame');
function point(e){const r=worldFrame.getBoundingClientRect();return{x:Math.max(0,Math.min(r.width,e.clientX-r.left)),y:Math.max(0,Math.min(r.height,e.clientY-r.top)),w:r.width,h:r.height}}
function drawBox(box){selection.style.display='block';selection.style.left=(box.x*100)+'%';selection.style.top=(box.y*100)+'%';selection.style.width=(box.width*100)+'%';selection.style.height=(box.height*100)+'%'}
function drawPrediction(raw,confidence){if(!raw||raw.length!==4||!worldFrame.naturalWidth){prediction.style.display='none';return}const w=worldFrame.naturalWidth,h=worldFrame.naturalHeight,r=16/9;let ox=0,oy=0,cw=w,ch=h;if(w/h>r){cw=h*r;ox=(w-cw)/2}else{ch=w/r;oy=(h-ch)/2}const x=(ox+raw[0]*cw)/w,y=(oy+raw[1]*ch)/h,x1=(ox+raw[2]*cw)/w,y1=(oy+raw[3]*ch)/h;prediction.style.display='block';prediction.style.left=(x*100)+'%';prediction.style.top=(y*100)+'%';prediction.style.width=(Math.max(0,x1-x)*100)+'%';prediction.style.height=(Math.max(0,y1-y)*100)+'%';prediction.title='ROCKET-2 learned target · '+Math.round((confidence||0)*100)+'%'}
viewer.onpointerdown=e=>{if(!worldFrame.complete)return;const p=point(e);dragging=true;startX=p.x;startY=p.y;viewer.setPointerCapture(e.pointerId);targetBox={x:p.x/p.w,y:p.y/p.h,width:.001,height:.001};drawBox(targetBox)};
viewer.onpointermove=e=>{if(!dragging)return;const p=point(e),x=Math.min(startX,p.x),y=Math.min(startY,p.y);targetBox={x:x/p.w,y:y/p.h,width:Math.max(1,Math.abs(p.x-startX))/p.w,height:Math.max(1,Math.abs(p.y-startY))/p.h};drawBox(targetBox)};
viewer.onpointerup=e=>{if(!dragging)return;dragging=false;viewer.releasePointerCapture(e.pointerId)};
function refreshFrame(){if(!dragging)worldFrame.src='/api/frame.png?t='+Date.now()}
async function refresh(){try{const s=await api('/api/status');$('dot').style.background='var(--green)';$('connection').textContent='Live telemetry';
const bi=s.bedrock.instances.length>0; $('bedrock').textContent=bi?'RUNNING':'STOPPED';cls($('bedrock'),bi);$('version').textContent=esc(s.bedrock.version);
const sup=esc(s.supervisor.state);$('supervisor').textContent=sup;cls($('supervisor'),s.supervisor_reachable&&sup!=='FAILSAFE');
const alive=s.agent.alive;$('agent').textContent=alive?'RUNNING':'DISARMED';cls($('agent'),alive);const t=s.telemetry||{};
$('skill').textContent=esc(t.active_skill);$('goal').textContent=esc(t.chosen_goal_id||'No active goal');$('reason').textContent=esc(t.reasoning_summary||'Waiting for agent telemetry.');
$('instruction').textContent=esc(t.active_instruction||'Waiting for an executable option.');const params=t.active_skill_parameters||{},paramText=Object.entries(params).map(([k,v])=>k+'='+v).join(' · ');$('constraints').textContent=paramText?'active contract · '+paramText:'no explicit action constraints';const recent=(t.recent_skill_runs||[])[0];$('outcome').textContent=recent?'last option · '+recent.skill_id+' · '+recent.outcome+(recent.failure_reason?' · '+recent.failure_reason:''):'no terminal option evidence yet';const p=t.policy||{},active=p.active_route==='grounded'?(p.grounded||p):(p.primary||p),pred=active.last_prediction||{},counts=active.learned_action_counts||{},suppressed=(counts['constraint_suppressed.attack']||0)+(counts['constraint_suppressed.use']||0)+(counts['constraint_suppressed.jump']||0);$('policy').textContent=esc(active.model_version||p.policy_id);$('policyMetrics').textContent=(active.last_inference_ms??'—')+' ms · '+esc(p.active_route||active.grounding_mode)+' · '+esc(active.accepted_predictions||0)+' learned · '+esc(counts.jump||0)+' jumps · '+esc(counts.camera||0)+' camera · '+suppressed+' constrained · target '+(pred.target_exists_probability==null?'—':Math.round(pred.target_exists_probability*100)+'%');
$('camera').textContent=esc(active.estimated_pitch_units??'—')+' / '+esc(active.camera_pitch_limit??'—');$('cameraMode').textContent=active.camera_recovery_active?'learned horizon recovery':'normal learned control';$('camera').className='value '+(active.camera_recovery_active?'amber':'ok');drawPrediction(pred.target_bbox_xyxy,pred.target_exists_probability);
const pf=((t.perception||{}).fresh_facts)||{};$('facts').textContent=Object.keys(pf).length?JSON.stringify(pf,null,2):'No fresh semantic facts yet.';
$('frames').textContent=esc(t.frames||0);$('actions').textContent=esc(t.motor_actions||0);$('capture').textContent=t.last_capture_ms==null?'—':t.last_capture_ms+' ms';
}catch(e){$('dot').style.background='var(--red)';$('connection').textContent='Disconnected'}}
async function messages(){try{const d=await api('/api/messages');const feed=$('feed');feed.replaceChildren();if(!d.messages.length){const x=document.createElement('div');x.className='label';x.textContent='No messages yet';feed.append(x);return}
d.messages.forEach(m=>{const box=document.createElement('div');box.className='msg';const meta=document.createElement('div');meta.className='meta';const k=document.createElement('span');k.className='kind';k.textContent=m.kind;const when=document.createElement('span');when.textContent=new Date(m.created_ns/1e6).toLocaleTimeString();const status=document.createElement('span');status.textContent=m.status;meta.append(k,when,status);const text=document.createElement('div');text.className='text';text.textContent=m.text;box.append(meta,text);if(m.response_text){const reply=document.createElement('div');reply.className='text ok';reply.textContent='Agent: '+m.response_text;box.append(reply)}feed.append(box)})}catch(e){}}
$('send').onclick=async()=>{const text=$('text').value.trim();if(!text)return;try{await api('/api/messages',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,kind:$('kind').value,priority:Number($('priority').value)})});$('text').value='';$('notice').textContent='Delivered to the durable cognition inbox.';messages()}catch(e){$('notice').textContent=e.message}};
$('setTarget').onclick=async()=>{if(!targetBox||targetBox.width<.005||targetBox.height<.005){$('targetNotice').textContent='Drag a non-empty target region first.';return}try{const t=await api('/api/target',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...targetBox,label:$('targetLabel').value.trim()||'target'})});$('targetNotice').textContent='ROCKET-2 target armed: '+t.label+' · '+t.track_id}catch(e){$('targetNotice').textContent=e.message}};
$('clearTarget').onclick=async()=>{try{await api('/api/target/clear',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});targetBox=null;selection.style.display='none';$('targetNotice').textContent='Grounded target cleared.'}catch(e){$('targetNotice').textContent=e.message}};
$('pause').onclick=async()=>{try{await api('/api/control/pause',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});$('notice').textContent='Agent paused; motor capability revoked.';refresh()}catch(e){$('notice').textContent=e.message}};
$('resume').onclick=async()=>{try{await api('/api/control/resume',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});$('notice').textContent='Supervisor returned to safe idle. Live control was not armed.';refresh()}catch(e){$('notice').textContent=e.message}};
refresh();messages();refreshFrame();setInterval(refresh,500);setInterval(messages,2000);setInterval(refreshFrame,1000);
</script></body></html>"""
