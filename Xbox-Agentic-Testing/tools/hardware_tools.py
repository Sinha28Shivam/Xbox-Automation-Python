"""
hardware_tools.py - probe the rig: GIMX, the capture card, the control config.

These are the health agent's eyes. They answer "can this rig run a test at all?"
and nothing else - none of them press a button or change console state, so they
are safe to run before every scenario and safe to grant to the RCA agent, which
must be able to look but not touch.

THE DISTINCTION THAT MATTERS
----------------------------
`check_gimx_session` reports REACHABLE, not WORKING. GIMX answers UDP as soon as
the server is up, whether or not the Guide-button handshake was done. Without
that handshake every event is accepted, reports "ok", and reaches nothing. This
project already lost time to exactly that, so the tool returns
`authenticated: null` and says so explicitly rather than implying success.
"""

from __future__ import annotations

from typing import Any

from registry import ToolContext, ToolSpec, fail, make_tool, ok


# ===========================================================================
# GIMX
# ===========================================================================
def _check_gimx_session(ctx: ToolContext) -> Any:
    def run() -> dict[str, Any]:
        try:
            g = ctx.hardware.gimx_functions()
        except Exception as exc:
            return fail(f"GIMX adapter unavailable: {exc}")

        try:
            pids = g["running_gimx_pids"]()
            reachable = g["session_is_up"](
                ctx.hardware.controls_path, None, True)
        except Exception as exc:
            return fail(f"Could not probe GIMX: {exc}")

        conn = ctx.hardware.controls.connection if _controls_ok(ctx) else {}
        return ok(
            processes=pids,
            process_running=bool(pids),
            reachable=bool(reachable),
            # Deliberately unknown. Only a visible screen change can prove the
            # session was authenticated, and that is the executor's job to
            # establish, not something we can assert from here.
            authenticated=None,
            udp_address=f"{conn.get('udp_address', '')}:{conn.get('udp_port', '')}",
            serial_port=conn.get("serial_port", ""),
            caveat=(
                "'reachable' means the GIMX server answered UDP. It does NOT "
                "mean the session was authenticated with the Guide button. "
                "Unauthenticated sessions accept every event and deliver none."
            ),
        )

    return make_tool(
        run, "check_gimx_session",
        "Probe the GIMX server: running processes, UDP reachability, serial "
        "port and UDP address. Returns reachable=true/false. Note that "
        "reachable does NOT prove the session is authenticated.")


def _restart_gimx_session(ctx: ToolContext) -> Any:
    def run() -> dict[str, Any]:
        if ctx.dry_run:
            return ok(skipped=True, reason="dry_run")
        try:
            g = ctx.hardware.gimx_functions()
            stopped = g["stop_all"](True)
        except Exception as exc:
            return fail(f"Could not stop GIMX: {exc}")
        return ok(
            stopped=bool(stopped),
            # We stop but never silently restart. A new session needs a human
            # to hold Guide for 2s; auto-starting one would leave an
            # unauthenticated server that looks healthy and works for nothing.
            requires_human=True,
            instructions=(
                "Start a session in its own terminal:\n"
                "  python gimx-session/gimx_session.py start\n"
                "then HOLD the controller's GUIDE button for 2 seconds."),
        )

    return make_tool(
        run, "restart_gimx_session",
        "Stop all running GIMX processes and release the serial port. Does "
        "NOT start a new session, because authentication needs a human to "
        "hold the Guide button.")


# ===========================================================================
# Capture card
# ===========================================================================
def _check_capture_device(ctx: ToolContext) -> Any:
    def run() -> dict[str, Any]:
        try:
            fns = ctx.hardware.capture_functions()
        except Exception as exc:
            return fail(f"Capture adapter unavailable: {exc}")

        try:
            pre = fns["preflight"](ctx.hardware.controls_path, True)
        except Exception as exc:
            return fail(f"Capture preflight failed: {exc}")

        identity = _check_device_identity(ctx)

        return ok(
            device_present=bool(pre.device_present),
            device_opens=bool(pre.device_opens),
            has_signal=bool(pre.has_signal),
            ready=bool(pre.ok),
            # Usually RECentral. Only one app may hold a UVC device, and the
            # resulting error ("device already in use") is easy to misread as a
            # hardware fault.
            conflicting_apps=list(pre.conflicts),
            messages=list(pre.messages),
            **identity,
        )

    return make_tool(
        run, "check_capture_device",
        "Preflight the capture card: is it connected, can it be opened (no "
        "other app holding it), and is a real video signal arriving? A flat "
        "frame means no HDMI signal, not a dark scene.")


# ===========================================================================
# Configuration introspection
# ===========================================================================
def _get_control_surface(ctx: ToolContext) -> Any:
    def run() -> dict[str, Any]:
        summary = ctx.hardware.controls_summary()
        if "error" in summary:
            return fail(str(summary["error"]))
        return ok(**summary)

    return make_tool(
        run, "get_control_surface",
        "List every control this rig supports - buttons, triggers, sticks, "
        "macros, special actions, console profiles and timings - read live "
        "from controls.yaml. Call this before planning so you only reference "
        "controls that actually exist.")


def _resolve_control(ctx: ToolContext) -> Any:
    def run(name: str) -> dict[str, Any]:
        try:
            kind, canonical = ctx.hardware.controls.resolve(name)
        except KeyError as exc:
            return fail(str(exc), requested=name, resolved=False)
        except Exception as exc:
            return fail(f"Could not resolve '{name}': {exc}")
        return ok(requested=name, kind=kind, canonical=canonical,
                  gimx=ctx.hardware.controls.gimx_name(name), resolved=True)

    return make_tool(
        run, "resolve_control",
        "Resolve a control name or alias (e.g. 'xbox', 'confirm', 'dpad_up') "
        "to its canonical name. Use it to check a scenario references real "
        "controls before executing anything.")


def _get_rig_status(ctx: ToolContext) -> Any:
    def run() -> dict[str, Any]:
        return ok(**ctx.hardware.status())

    return make_tool(
        run, "get_rig_status",
        "Report which hardware adapters loaded successfully and which failed, "
        "with the reason. Use this first when something behaves oddly - a "
        "missing adapter explains a lot of downstream confusion.")


def _check_device_identity(ctx: ToolContext) -> dict[str, Any]:
    """Are we looking at the capture card, or at something that resembles one?

    This exists because of a real failure. The OpenCV indices swapped, the
    framework opened the laptop webcam instead of the card, and reported
    "black screen / no HDMI signal" - while the Xbox was fine on the other
    index. A dim webcam is a *plausible* capture device: it opens, it returns
    frames, and those frames are dark, so a plain is-it-blank check agrees.

    Frame SIZE is what separates them. The card delivers the configured
    resolution; the webcam delivered 1280x720 against a configured 1920x1080.
    """
    out: dict[str, Any] = {"device_identity_ok": True, "device_warning": None}
    try:
        cam = ctx.hardware.capture()
        controls = ctx.hardware.controls
        capture_cfg = (controls.data.get("capture") or {}
                       if hasattr(controls, "data") else {})
        want_w = int(capture_cfg.get("width", 1920))
        want_h = int(capture_cfg.get("height", 1080))

        frame = cam.grab(allow_blank=True)
        if frame is None:
            return out

        h, w = frame.shape[:2]
        out["frame_size"] = f"{w}x{h}"
        out["expected_size"] = f"{want_w}x{want_h}"
        out["resolved_index"] = getattr(cam, "index", None)
        out["index_source"] = getattr(cam, "index_source", "")

        if (w, h) != (want_w, want_h):
            out["device_identity_ok"] = False
            out["device_warning"] = (
                f"Frame is {w}x{h} but {want_w}x{want_h} was configured. "
                f"This often means the WRONG DEVICE is open - a webcam rather "
                f"than the capture card. OpenCV indices shift when USB devices "
                f"change. Run _diagnose_capture.py to see a frame from every "
                f"device. (device resolved by: "
                f"{getattr(cam, 'index_source', 'unknown')})")
    except Exception:
        # Identity checking is a bonus; never let it break the health probe.
        pass
    return out


def _controls_ok(ctx: ToolContext) -> bool:
    try:
        _ = ctx.hardware.controls
        return True
    except Exception:
        return False


# ===========================================================================
# Registration
# ===========================================================================
def provide() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="check_gimx_session",
            description="Probe the GIMX server and serial link.",
            tags=["hardware", "diagnostic"],
            factory=_check_gimx_session,
        ),
        ToolSpec(
            name="restart_gimx_session",
            description="Stop GIMX and release the serial port.",
            tags=["hardware", "recovery"],
            factory=_restart_gimx_session,
            mutates_hardware=True,
        ),
        ToolSpec(
            name="check_capture_device",
            description="Preflight the capture card and HDMI signal.",
            tags=["hardware", "diagnostic", "vision"],
            factory=_check_capture_device,
        ),
        ToolSpec(
            name="get_control_surface",
            description="Every control available, read from controls.yaml.",
            tags=["introspection", "diagnostic"],
            factory=_get_control_surface,
        ),
        ToolSpec(
            name="resolve_control",
            description="Resolve a control name/alias to its canonical form.",
            tags=["introspection"],
            factory=_resolve_control,
        ),
        ToolSpec(
            name="get_rig_status",
            description="Adapter load status and configured paths.",
            tags=["diagnostic", "introspection"],
            factory=_get_rig_status,
        ),
    ]
