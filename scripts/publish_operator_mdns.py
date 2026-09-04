"""Keep the operator's existing computer-name.local reachable on the private LAN."""

from __future__ import annotations

import ipaddress
import json
from pathlib import Path
import signal
import socket
import subprocess
import threading


def lan_address() -> str:
    """Select the private address of a physical interface with a default route."""
    routes = json.loads(
        subprocess.check_output(["ip", "-j", "-4", "route", "show", "default"], text=True)
    )
    for route in sorted(routes, key=lambda item: item.get("metric", 0)):
        interface = route.get("dev", "")
        if not interface or not Path("/sys/class/net", interface, "device").exists():
            continue
        interfaces = json.loads(
            subprocess.check_output(["ip", "-j", "-4", "addr", "show", "dev", interface], text=True)
        )
        for address in interfaces[0].get("addr_info", []) if interfaces else []:
            candidate = address.get("local", "")
            if address.get("scope") != "global":
                continue
            parsed = ipaddress.IPv4Address(candidate)
            if any(
                parsed in ipaddress.IPv4Network(network)
                for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
            ):
                return str(parsed)
    raise RuntimeError("No private IPv4 address on a physical default-route interface")


def stop_publisher(publisher: subprocess.Popen[bytes] | None) -> None:
    if publisher is None or publisher.poll() is not None:
        return
    publisher.terminate()
    try:
        publisher.wait(timeout=5)
    except subprocess.TimeoutExpired:
        publisher.kill()
        publisher.wait()


def main() -> int:
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    publisher: subprocess.Popen[bytes] | None = None
    published: tuple[str, str] | None = None
    try:
        while not stopped.is_set():
            current = (socket.gethostname().split(".", 1)[0] + ".local", lan_address())
            if publisher is not None and publisher.poll() is not None:
                return 1
            if current != published:
                stop_publisher(publisher)
                # Existing aliases may own the reverse record. Publish only the A
                # record so this hostname can coexist without renaming anything.
                publisher = subprocess.Popen(["avahi-publish-address", "--no-reverse", *current])
                published = current
                print(f"Advertising {current[0]} at {current[1]}", flush=True)
            stopped.wait(30)
    finally:
        stop_publisher(publisher)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
