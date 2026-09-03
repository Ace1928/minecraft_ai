#!/usr/bin/python3
"""System-Python helper for GNOME Mutter ScreenCast frame acquisition.

This module deliberately uses only distribution-provided GI/dbus bindings and
the standard library.  Pixel data is written to shared memory; stdout is a
strict JSON-lines control protocol consumed by ``MutterPipeWireCapture``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import shared_memory
from typing import Any, cast


def _reply(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--monitor-width", required=True, type=int)
    parser.add_argument("--monitor-height", required=True, type=int)
    parser.add_argument("--crop-x", required=True, type=int)
    parser.add_argument("--crop-y", required=True, type=int)
    parser.add_argument("--crop-width", required=True, type=int)
    parser.add_argument("--crop-height", required=True, type=int)
    return parser


class _Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        import dbus  # type: ignore[import-not-found]
        import dbus.mainloop.glib  # type: ignore[import-not-found]
        import gi  # type: ignore[import-not-found]

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import GLib, Gst, GstVideo  # type: ignore[import-not-found]

        self._dbus = dbus
        self._glib = GLib
        self._gst = Gst
        self._gst_video = GstVideo
        self._pipeline: Any | None = None
        self._session: Any | None = None
        self._shared: shared_memory.SharedMemory | None = None
        self._frame_id = 0
        self._monitor_width = int(args.monitor_width)
        self._monitor_height = int(args.monitor_height)
        self._crop = (
            int(args.crop_x),
            int(args.crop_y),
            int(args.crop_width),
            int(args.crop_height),
        )
        self._validate_geometry()
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        Gst.init(None)
        self._start_screencast(str(args.output))

    def _validate_geometry(self) -> None:
        x, y, width, height = self._crop
        if self._monitor_width <= 0 or self._monitor_height <= 0:
            raise RuntimeError("invalid monitor dimensions")
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise RuntimeError("invalid capture crop")
        if x + width > self._monitor_width or y + height > self._monitor_height:
            raise RuntimeError("capture crop exceeds monitor")

    def _start_screencast(self, output: str) -> None:
        dbus = self._dbus
        bus = dbus.SessionBus()
        service = bus.get_object(
            "org.gnome.Mutter.ScreenCast",
            "/org/gnome/Mutter/ScreenCast",
        )
        screen_cast = dbus.Interface(service, "org.gnome.Mutter.ScreenCast")
        session_path = screen_cast.CreateSession(dbus.Dictionary({}, signature="sv"), timeout=5)
        session_object = bus.get_object("org.gnome.Mutter.ScreenCast", session_path)
        self._session = dbus.Interface(
            session_object,
            "org.gnome.Mutter.ScreenCast.Session",
        )
        stream_path = self._session.RecordMonitor(
            output,
            dbus.Dictionary({"cursor-mode": dbus.UInt32(0)}, signature="sv"),
            timeout=5,
        )
        stream_object = bus.get_object("org.gnome.Mutter.ScreenCast", stream_path)
        node: list[int] = []

        def _node_added(node_id: object) -> None:
            node.append(int(cast(Any, node_id)))

        stream_object.connect_to_signal(
            "PipeWireStreamAdded",
            _node_added,
            dbus_interface="org.gnome.Mutter.ScreenCast.Stream",
        )
        self._session.Start(timeout=5)
        deadline = time.monotonic() + 5.0
        context = self._glib.MainContext.default()
        while not node and time.monotonic() < deadline:
            while context.pending():
                context.iteration(False)
            time.sleep(0.005)
        if not node:
            raise RuntimeError("Mutter did not publish a PipeWire stream node")
        description = (
            f"pipewiresrc path={node[0]} do-timestamp=true ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "videoconvert ! video/x-raw,format=BGRA ! "
            "appsink name=capture_sink max-buffers=1 drop=true sync=false"
        )
        self._pipeline = self._gst.parse_launch(description)
        result = self._pipeline.set_state(self._gst.State.PLAYING)
        if result == self._gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer failed to start PipeWire pipeline")
        sample = self._pull_sample(timeout_ms=3000)
        width, height, _stride = self._sample_geometry(sample)
        if (width, height) != (self._monitor_width, self._monitor_height):
            raise RuntimeError(
                f"Mutter stream is {width}x{height}, expected "
                f"{self._monitor_width}x{self._monitor_height}"
            )
        _x, _y, crop_width, crop_height = self._crop
        size = crop_width * crop_height * 4
        self._shared = shared_memory.SharedMemory(create=True, size=size)
        _reply(
            {
                "ok": True,
                "event": "ready",
                "shm_name": self._shared.name,
                "width": crop_width,
                "height": crop_height,
                "size": size,
            }
        )

    def _pull_sample(self, *, timeout_ms: int) -> Any:
        if self._pipeline is None:
            raise RuntimeError("GStreamer pipeline is unavailable")
        sink = self._pipeline.get_by_name("capture_sink")
        if sink is None:
            raise RuntimeError("GStreamer appsink is unavailable")
        sample = sink.emit("try-pull-sample", int(timeout_ms) * 1_000_000)
        if sample is None:
            raise RuntimeError("PipeWire frame timed out")
        return sample

    def _sample_geometry(self, sample: Any) -> tuple[int, int, int]:
        caps = sample.get_caps()
        if caps is None:
            raise RuntimeError("PipeWire frame has no video caps")
        info = self._gst_video.VideoInfo.new_from_caps(caps)
        width = int(info.width)
        height = int(info.height)
        stride = int(info.stride[0])
        if width <= 0 or height <= 0 or stride < width * 4:
            raise RuntimeError("PipeWire frame has invalid BGRA layout")
        return width, height, stride

    def capture(self, timeout_ms: int) -> None:
        sample = self._pull_sample(timeout_ms=timeout_ms)
        width, height, stride = self._sample_geometry(sample)
        if (width, height) != (self._monitor_width, self._monitor_height):
            raise RuntimeError("PipeWire monitor geometry changed")
        buffer = sample.get_buffer()
        if buffer is None:
            raise RuntimeError("PipeWire sample has no buffer")
        mapped, mapping = buffer.map(self._gst.MapFlags.READ)
        if not mapped:
            raise RuntimeError("cannot map PipeWire frame")
        try:
            x, y, crop_width, crop_height = self._crop
            row_bytes = crop_width * 4
            required = (y + crop_height - 1) * stride + (x * 4) + row_bytes
            if int(mapping.size) < required:
                raise RuntimeError("PipeWire frame buffer is truncated")
            block = self._shared
            if block is None or block.size < row_bytes * crop_height:
                raise RuntimeError("capture shared memory is unavailable")
            source = memoryview(mapping.data)
            destination = block.buf
            if destination is None:
                raise RuntimeError("capture shared-memory buffer is unavailable")
            for row in range(crop_height):
                source_start = (y + row) * stride + x * 4
                destination_start = row * row_bytes
                destination[destination_start : destination_start + row_bytes] = source[
                    source_start : source_start + row_bytes
                ]
            source.release()
        finally:
            buffer.unmap(mapping)
        self._frame_id += 1
        _reply(
            {
                "ok": True,
                "event": "frame",
                "frame_id": self._frame_id,
                "captured_ns": time.monotonic_ns(),
                "bytes": self._crop[2] * self._crop[3] * 4,
            }
        )

    def run(self) -> None:
        for line in sys.stdin:
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise RuntimeError("command must be an object")
                command = payload.get("command")
                if command == "close":
                    return
                if command != "capture":
                    raise RuntimeError("unknown capture command")
                timeout_ms = payload.get("timeout_ms")
                if (
                    not isinstance(timeout_ms, int)
                    or isinstance(timeout_ms, bool)
                    or timeout_ms <= 0
                    or timeout_ms > 5000
                ):
                    raise RuntimeError("invalid capture timeout")
                self.capture(timeout_ms)
            except Exception as exc:
                _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
            self._pipeline = None
        if self._session is not None:
            try:
                self._session.Stop(timeout=2)
            except Exception:
                pass
            self._session = None
        if self._shared is not None:
            self._shared.close()
            try:
                self._shared.unlink()
            except FileNotFoundError:
                pass
            self._shared = None


def main() -> int:
    worker: _Worker | None = None
    try:
        worker = _Worker(_parser().parse_args())
        worker.run()
        return 0
    except Exception as exc:
        _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
