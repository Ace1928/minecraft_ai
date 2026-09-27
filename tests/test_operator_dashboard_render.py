"""Exercise the actual refresh function with a tiny DOM, no browser or network."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from minecraft_ai.operator.server import DASHBOARD_HTML


def test_current_values_are_cleared_on_stale_response_and_http_failure():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the isolated dashboard JavaScript test")
    clear = DASHBOARD_HTML.split("function clearTelemetryPanels(){", 1)[1].split(
        "\nlet dragging=", 1,
    )[0]
    refresh = DASHBOARD_HTML.split("async function refresh(){", 1)[1].split(
        "\nasync function messages()", 1,
    )[0]
    script = r"""
const assert=require('node:assert/strict');
const document={hidden:false};
const elements=new Map();
const $=id=>{
 if(!elements.has(id))elements.set(id,{textContent:'',className:'',style:{}});
 return elements.get(id)
};
const esc=v=>v??'—';
const cls=(el,good)=>{el.className='value '+(good?'ok':'bad')};
const drawPrediction=()=>{};
let statusLoading=false,fail=false;
let response={telemetry_current:true,bedrock:{instances:['fixture'],version:'fixture-version'},
  supervisor_reachable:true,supervisor:{state:'RUNNING',world_camera:{origin_calibrated:true}},
  agent:{alive:true},telemetry:{frames:1324,motor_actions:77,last_capture_ms:45,
    active_skill:'fixture-skill',reasoning_summary:'fixture-reason',
    perception:{fresh_facts:{fixture_fact:true}},
    trajectory_recording:{enabled:true,written_steps:55}}};
const api=async()=>{if(fail)throw Error('fixture HTTP 503');return response};
"""
    script += "function clearTelemetryPanels(){" + clear
    script += "\nasync function refresh(){" + refresh
    script += r"""
(async()=>{
 await refresh();
 assert.equal($('skill').textContent,'fixture-skill');
 assert.equal($('frames').textContent,1324);
 assert.equal($('recording').textContent,'ON');
 response.telemetry_current=false;
 await refresh();
 for(const id of ['skill','frames','actions','capture','recording','recordingDetail'])
   assert.equal($(id).textContent,'Unavailable');
 assert.equal($('supervisor').textContent,'RUNNING');
 assert.equal($('bedrock').textContent,'RUNNING');
 assert.ok(!$('facts').textContent.includes('fixture_fact'));
 assert.equal($('prediction').style.display,'none');
 response.telemetry_current=true;
 await refresh();
 assert.equal($('frames').textContent,1324);
 fail=true;
 await refresh();
 assert.equal($('connection').textContent,'Disconnected');
 for(const id of ['skill','frames','actions','capture','recording','recordingDetail'])
   assert.equal($(id).textContent,'Unavailable');
 for(const id of ['supervisor','agent','bedrock','camera'])
   assert.equal($(id).textContent,'Unconfirmed');
 assert.ok(!$('facts').textContent.includes('fixture_fact'));
 assert.equal(statusLoading,false);
 fail=false;
 await refresh();
 assert.equal($('connection').textContent,'Live agent telemetry');
 assert.equal($('frames').textContent,1324);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, json.dumps({"stdout": result.stdout, "stderr": result.stderr})


def test_topology_fetch_failure_repaints_cleared_state_instead_of_retaining_old_pixels():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the isolated dashboard JavaScript test")
    refresh = DASHBOARD_HTML.split("async function refreshTopology(){", 1)[1].split(
        "\nfunction drawSystemTopology()", 1,
    )[0]
    script = r"""
const assert=require('node:assert/strict');
const document={hidden:false};
let topology={parts:['previous']},topoLoading=false;
const rendered=[];
const api=async()=>{throw Error('fixture HTTP503')};
const drawSystemTopology=()=>rendered.push(['system',topology]);
const drawBrain=()=>rendered.push(['brain',topology]);
"""
    script += "async function refreshTopology(){" + refresh
    script += r"""
(async()=>{
 await refreshTopology();
 assert.deepEqual(rendered,[['system',null],['brain',null]]);
 assert.equal(topoLoading,false);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_hidden_dashboard_suppresses_every_polling_family_before_network_or_decode():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the isolated dashboard JavaScript test")
    functions = [
        ("refreshFrame", "\nasync function refresh()"),
        ("refresh", "\nasync function messages()"),
        ("messages", "\nasync function setStandby"),
        ("inventoryStatus", "\n$('inventoryCheck').onclick"),
        ("refreshTopology", "\nfunction drawSystemTopology()"),
    ]
    script = """
const assert=require('node:assert/strict');const document={hidden:true};
let statusLoading=false,messagesLoading=false,frameLoading=false,topoLoading=false;
let inventoryLoading=false,inventorySubmitting=false,dragging=false,targetBox=null;
let networkCalls=0;
const api=async()=>{networkCalls++;throw Error('hidden network')};
const fetch=api;
const $=()=>{throw Error('hidden DOM processing')};
"""
    for name, end in functions:
        body = DASHBOARD_HTML.split(f"async function {name}(){{", 1)[1].split(end, 1)[0]
        script += f"\nasync function {name}(){{" + body
    script += """
(async()=>{
 await refreshFrame();await refresh();await messages();
 await inventoryStatus();await refreshTopology();
 assert.equal(networkCalls,0);
 assert.equal(statusLoading||messagesLoading||frameLoading||topoLoading||inventoryLoading,false);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
