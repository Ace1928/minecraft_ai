"""Check the generic Wine patch without loading Wine, Xlib, or the live game."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/winegdk/virtual-desktop-clip-owner.patch"
FIXTURES = Path(__file__).parent / "fixtures/wine_clip_focus"
SIGNATURE = "static BOOL clipping_focus_allows_grab( struct x11drv_thread_data *data )"


def _extract_predicate(patch: str) -> str:
    """Compile the actual added function, never a test-only implementation."""
    lines = patch.splitlines()
    starts = [index for index, line in enumerate(lines) if line == "+" + SIGNATURE]
    assert len(starts) == 1, "expected exactly one added clipping predicate"
    body: list[str] = []
    for line in lines[starts[0]:]:
        assert line.startswith("+"), "clipping predicate crossed a patch hunk boundary"
        body.append(line[1:])
        if line == "+}":
            return "\n".join(body) + "\n"
    raise AssertionError("unterminated clipping predicate")


def test_patch_has_one_clipping_local_predicate_and_preserves_callsite_order() -> None:
    patch = PATCH.read_text(encoding="utf-8")
    assert [line for line in patch.splitlines() if line.startswith("diff --git ")] == [
        "diff --git a/dlls/winex11.drv/mouse.c b/dlls/winex11.drv/mouse.c",
    ]
    predicate = _extract_predicate(patch)
    assert predicate.count("XGetInputFocus(") == 1
    assert "if (!XGetInputFocus( data->display, &focus, &revert )) return FALSE;" in predicate
    assert "is_current_process_focused" not in predicate
    assert predicate.index("if (!data || !data->display)") < predicate.index("XGetInputFocus(")
    assert "+#ifdef HAVE_X11_EXTENSIONS_XINPUT2_H\n" in patch
    assert patch.index("/* don't clip in the desktop process */") < patch.index(
        "+    if (!data) return FALSE;"
    ) < patch.index("+    if (!clipping_focus_allows_grab( data )) return TRUE;")
    assert "keyboard_grabbed" not in patch


def test_exact_patch_predicate_passes_native_stub_checks(tmp_path: Path) -> None:
    predicate = _extract_predicate(PATCH.read_text(encoding="utf-8"))
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        pytest.skip("native C compiler unavailable; source extraction is tested separately")
    source = (
        (FIXTURES / "prefix.c").read_text(encoding="utf-8")
        + "\n" + predicate + "\n"
        + (FIXTURES / "suffix.c").read_text(encoding="utf-8")
    )
    executable = tmp_path / ("clip-focus.exe" if os.name == "nt" else "clip-focus")
    build = subprocess.run(
        [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2", "-x", "c", "-",
         "-o", str(executable)],
        input=source, text=True, capture_output=True, check=False, timeout=60,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run(
        [str(executable)], text=True, capture_output=True, check=False, timeout=5,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len([line for line in result.stdout.splitlines() if line.startswith("PASS ")]) == 15
    assert result.stdout.endswith(
        "15 checks passed; predicate and modeled boundary only, not driver integration.\n"
    )
