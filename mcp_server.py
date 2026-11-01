"""MCP Server for VeoHS / Flowhs (Muse + Google Flow + PureBase Watermark Remover).

Exposes tools via Model Context Protocol (stdio / SSE) so AI Agents can:
- Generate images via Muse (`muse-image`) or Google Flow (`NANO_BANANA_PRO`, `BELUGA`, `HARBOR_SEAL`)
- Generate videos (T2V & I2V) via Muse (`muse-video`) or Google Flow (`abra_i2v_8s` / Omni 1.1 Flash, `Veo 3.1`)
- Inspect & manage Muse accounts, quotas (`1B tokens`), and Google Flow Chrome Extension profiles
- Clean image/video watermarks in PureBase mode (`gwr.mjs`)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

FLOWHS_DIR = Path(__file__).resolve().parent
if str(FLOWHS_DIR) not in sys.path:
    sys.path.insert(0, str(FLOWHS_DIR))

import httpx
from mcp.server.fastmcp import FastMCP

from muse2api_client import (
    DEFAULT_MUSE2API_KEY,
    DEFAULT_MUSE2API_URL,
    Muse2APIClient,
    auto_login_muse_account,
    ensure_local_muse2api_running,
    generate_image_muse,
    generate_video_muse,
    parse_muse_cookies,
    verify_us_proxy,
)

mcp = FastMCP("flowhs")

BRIDGE_HTTP_URL = os.environ.get("FLOWHS_BRIDGE_HTTP_URL", "http://127.0.0.1:8100")
DEFAULT_OUT_DIR = FLOWHS_DIR / "outputs" / "mcp"
_LOCAL_BRIDGE = None


def _load_flowhs_config() -> dict[str, Any]:
    cfg_path = FLOWHS_DIR / "config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _get_muse_settings() -> tuple[str, str, str]:
    cfg = _load_flowhs_config()
    base_url = cfg.get("muse2api_url") or DEFAULT_MUSE2API_URL
    api_key = cfg.get("muse2api_key") or DEFAULT_MUSE2API_KEY
    proxy = cfg.get("muse2api_proxy") or cfg.get("muse_proxy") or ""
    return base_url, api_key, proxy


async def _clean_media_watermark(file_path: Path) -> str:
    """Run PureBase watermark removal on an image or video file."""
    if not file_path.exists():
        return str(file_path)
    try:
        from flowzero_remover import remove_image_watermark, remove_video_watermark
        ext = file_path.suffix.lower()
        tmp_out = file_path.with_name(f"{file_path.stem}_clean{ext}")
        loop = asyncio.get_running_loop()
        if ext in (".png", ".jpg", ".jpeg", ".webp"):
            ok = await loop.run_in_executor(
                None,
                lambda: remove_image_watermark(str(file_path), str(tmp_out), mode="PureBase"),
            )
        elif ext in (".mp4", ".mov", ".webm"):
            ok = await loop.run_in_executor(
                None,
                lambda: remove_video_watermark(str(file_path), str(tmp_out), mode="PureBase"),
            )
        else:
            ok = False
        if ok and tmp_out.exists() and tmp_out.stat().st_size > 0:
            tmp_out.replace(file_path)
    except Exception:
        pass
    return str(file_path.resolve())


async def _call_running_bridge(path: str, method: str = "GET", payload: dict | None = None, timeout: float = 360.0) -> dict[str, Any] | None:
    """Talk to the running Flowhs GUI bridge on 127.0.0.1:8100 if available."""
    url = f"{BRIDGE_HTTP_URL.rstrip('/')}{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as cl:
            if method == "GET":
                r = await cl.get(url)
            else:
                r = await cl.post(url, json=payload or {})
            if r.status_code == 200:
                return r.json()
    except Exception:
        return None
    return None


async def _get_or_start_local_bridge():
    global _LOCAL_BRIDGE
    if _LOCAL_BRIDGE is not None:
        return _LOCAL_BRIDGE
    from bridge import FlowBridge
    b = FlowBridge()
    await b.start()
    _LOCAL_BRIDGE = b
    return b


@mcp.tool()
async def flowhs_status() -> dict[str, Any]:
    """Check readiness and active accounts of Muse (`muse2api`) and Google Flow (`FlowBridge`)."""
    base_url, api_key, proxy = _get_muse_settings()
    await ensure_local_muse2api_running(base_url, proxy=proxy)
    muse_cl = Muse2APIClient(base_url, api_key)
    try:
        muse_ready = await muse_cl.readyz()
        muse_accs = (await muse_cl.list_accounts()).get("data", [])
    except Exception as e:
        muse_ready = {"ready": False, "error": str(e)}
        muse_accs = []

    flow_status = await _call_running_bridge("/api/bridge/status", method="GET", timeout=5.0)
    if flow_status is None:
        flow_status = {"ok": False, "note": "Flowhs GUI bridge (port 8100) is not running or using older process"}

    return {
        "muse": {
            "base_url": base_url,
            "readyz": muse_ready,
            "accounts": [
                {
                    "id": a.get("id"),
                    "label": a.get("label"),
                    "status": a.get("status"),
                    "enabled": a.get("enabled"),
                    "success_count": a.get("success_count"),
                    "fail_count": a.get("fail_count"),
                }
                for a in muse_accs
            ],
        },
        "google_flow": flow_status,
    }


@mcp.tool()
async def flowhs_generate_image(
    prompt: str,
    output_path: str = "",
    engine: str = "muse",
    model: str = "muse-image",
    aspect: str = "16:9",
    reference_images: list[str] | None = None,
    clean_watermark: bool = True,
) -> dict[str, Any]:
    """Generate an AI image using Muse (`muse-image`) or Google Flow (`NANO_BANANA_PRO`, `BELUGA`, `HARBOR_SEAL`).

    Args:
        prompt: Detailed visual prompt for the image.
        output_path: File path to save the generated PNG image (auto-generated in `Flowhs/outputs/mcp/` if empty).
        engine: `"muse"` (default), `"flow"`, or `"auto"` (tries Muse first, falls back to Google Flow).
        model: Model name (`"muse-image"`, `"NANO_BANANA_PRO"`, `"BELUGA"`, `"HARBOR_SEAL"`).
        aspect: Aspect ratio (`"16:9"`, `"9:16"`, `"1:1"`, `"4:3"`, `"3:4"`).
        reference_images: Optional list of up to 4 reference image file paths or URLs for character/style consistency.
        clean_watermark: Whether to automatically strip watermarks in PureBase mode.
    """
    out_file = Path(output_path) if output_path else (DEFAULT_OUT_DIR / f"img_{int(time.time())}_{uuid.uuid4().hex[:6]}.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    eng = (engine or "muse").strip().lower()
    if model.upper() in ("NANO_BANANA_PRO", "BELUGA", "HARBOR_SEAL", "GEM_PIX_2") and eng == "muse":
        eng = "flow"

    base_url, api_key, proxy = _get_muse_settings()
    errors = []

    if eng in ("muse", "auto"):
        try:
            await ensure_local_muse2api_running(base_url, proxy=proxy)
            res = await generate_image_muse(
                prompt=prompt,
                aspect=aspect,
                out_path=out_file,
                reference_images=reference_images,
                base_url=base_url,
                api_key=api_key,
            )
            if clean_watermark and out_file.exists():
                await _clean_media_watermark(out_file)
            return {
                "ok": True,
                "engine": "muse",
                "model": "muse-image",
                "saved_path": str(out_file.resolve()),
                "url": res.get("url"),
                "media_id": res.get("media_id"),
            }
        except Exception as e:
            errors.append(f"Muse error: {e}")
            if eng == "muse":
                return {"ok": False, "engine": "muse", "error": str(e)}

    # Google Flow path
    flow_model = model if model.upper() not in ("MUSE-IMAGE", "MUSE_IMAGE", "MUSE") else "NANO_BANANA_PRO"
    bridge_resp = await _call_running_bridge(
        "/api/bridge/generate_image",
        method="POST",
        payload={
            "prompt": prompt,
            "model": flow_model,
            "aspect": aspect,
            "reference_images": reference_images or [],
            "out_path": str(out_file.resolve()),
        },
        timeout=300.0,
    )
    if bridge_resp is None:
        b = await _get_or_start_local_bridge()
        r = await b.generate_image(prompt=prompt, project_id="", aspect=aspect, model=flow_model)
        if r and r.url:
            await b.download(r.url, out_file)
            bridge_resp = {"ok": True, "media_id": r.media_id, "url": r.url, "saved_path": str(out_file.resolve())}
        else:
            bridge_resp = {"ok": False, "error": getattr(r, "error", "Flow generate_image failed")}

    if bridge_resp.get("ok") and out_file.exists() and clean_watermark:
        await _clean_media_watermark(out_file)
    bridge_resp["engine"] = "flow"
    if errors:
        bridge_resp["fallback_notes"] = errors
    return bridge_resp


@mcp.tool()
async def flowhs_generate_video(
    prompt: str,
    output_path: str = "",
    image_path: str = "",
    engine: str = "muse",
    model: str = "muse-video",
    aspect: str = "16:9",
    seconds: int = 8,
    clean_watermark: bool = True,
) -> dict[str, Any]:
    """Generate an AI video (Text-to-Video or Image-to-Video) using Muse (`muse-video`) or Google Flow (`abra_i2v_8s` / Omni 1.1 Flash, `Veo 3.1`).

    Args:
        prompt: Motion / scene prompt for the video.
        output_path: File path to save the `.mp4` video (auto-generated in `Flowhs/outputs/mcp/` if empty).
        image_path: Optional start frame image path or URL for Image-to-Video (leave empty for Text-to-Video).
        engine: `"muse"` (default), `"flow"`, or `"auto"` (tries Muse first, falls back to Google Flow).
        model: `"muse-video"`, `"omni_flash"` (`abra_i2v_8s`), `"lite_low"` (Veo 3.1 Lite), or `"s_fast"` (Veo 3.1 Fast).
        aspect: `"16:9"` or `"9:16"`.
        seconds: Video duration in seconds (default `8`).
        clean_watermark: Whether to automatically strip watermarks in PureBase mode.
    """
    out_file = Path(output_path) if output_path else (DEFAULT_OUT_DIR / f"vid_{int(time.time())}_{uuid.uuid4().hex[:6]}.mp4")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    eng = (engine or "muse").strip().lower()
    if model.lower() in ("omni_flash", "abra_i2v_8s", "lite", "lite_low", "fast", "s_fast") and eng == "muse":
        eng = "flow"

    base_url, api_key, proxy = _get_muse_settings()
    errors = []

    if eng in ("muse", "auto"):
        try:
            await ensure_local_muse2api_running(base_url, proxy=proxy)
            res = await generate_video_muse(
                image_source=image_path if image_path else None,
                prompt=prompt,
                aspect=aspect,
                out_path=out_file,
                seconds=seconds,
                base_url=base_url,
                api_key=api_key,
            )
            if clean_watermark and out_file.exists():
                await _clean_media_watermark(out_file)
            return {
                "ok": True,
                "engine": "muse",
                "model": "muse-video",
                "saved_path": str(out_file.resolve()),
                "url": res.get("url"),
                "media_id": res.get("media_id"),
            }
        except Exception as e:
            errors.append(f"Muse video error: {e}")
            if eng == "muse":
                return {"ok": False, "engine": "muse", "error": str(e)}

    # Google Flow path
    flow_model = model if model.lower() not in ("muse-video", "muse_video", "muse") else "abra_i2v_8s"
    if flow_model == "omni_flash":
        flow_model = "abra_i2v_8s"
    bridge_resp = await _call_running_bridge(
        "/api/bridge/generate_video",
        method="POST",
        payload={
            "prompt": prompt,
            "model": flow_model,
            "aspect": aspect,
            "image_path": image_path or None,
            "out_path": str(out_file.resolve()),
        },
        timeout=600.0,
    )
    if bridge_resp is None:
        return {
            "ok": False,
            "engine": "flow",
            "error": "Flowhs bridge HTTP API not reachable on port 8100 (restart Flowhs to enable /api/bridge/*)",
            "fallback_notes": errors,
        }
    if bridge_resp.get("ok") and out_file.exists() and clean_watermark:
        await _clean_media_watermark(out_file)
    bridge_resp["engine"] = "flow"
    if errors:
        bridge_resp["fallback_notes"] = errors
    return bridge_resp


@mcp.tool()
async def flowhs_muse_quota(refresh: bool = False) -> dict[str, Any]:
    """Query weekly usage percentage and remaining extra tokens (`1B tokens left`) for all Muse accounts."""
    base_url, api_key, proxy = _get_muse_settings()
    await ensure_local_muse2api_running(base_url, proxy=proxy)
    cl = Muse2APIClient(base_url, api_key)
    accs = (await cl.list_accounts()).get("data", [])
    results = []
    q_suffix = "?refresh=1" if refresh else ""
    for a in accs:
        aid = a.get("id")
        try:
            q = await cl._request("GET", f"/admin/accounts/{aid}/quota{q_suffix}", timeout=45.0)
            results.append({"id": aid, "label": a.get("label"), "quota": q.get("quota")})
        except Exception as e:
            results.append({"id": aid, "label": a.get("label"), "error": str(e)})
    return {"accounts": results}


@mcp.tool()
async def flowhs_muse_accounts(
    action: str = "list",
    label: str = "",
    cookies_raw: str = "",
    account_id: str = "",
) -> dict[str, Any]:
    """Manage Muse accounts in `muse2api`.

    Args:
        action: `"list"`, `"add"`, `"reset"`, `"renew"`, `"delete"`, or `"sync_extension"`.
        label: Account label/email when `action="add"`.
        cookies_raw: Cookie JSON or header string when `action="add"`.
        account_id: Target account ID (`acc_...`) for `"reset"`, `"renew"`, or `"delete"`.
    """
    base_url, api_key, proxy = _get_muse_settings()
    await ensure_local_muse2api_running(base_url, proxy=proxy)
    cl = Muse2APIClient(base_url, api_key)
    act = (action or "list").strip().lower()

    if act == "list":
        return await cl.list_accounts()
    if act == "add":
        cookies = parse_muse_cookies(cookies_raw)
        return await cl.add_account(label or "muse_acc", cookies)
    if act == "reset" and account_id:
        return await cl._request("POST", f"/admin/accounts/{account_id}/reset")
    if act == "renew" and account_id:
        return await cl._request("POST", f"/admin/accounts/{account_id}/renew")
    if act == "delete" and account_id:
        return await cl._request("DELETE", f"/admin/accounts/{account_id}")
    if act == "sync_extension":
        resp = await _call_running_bridge("/api/bridge/muse_cookies", method="POST", timeout=20.0)
        items = (resp or {}).get("items") or []
        imported = []
        for it in items:
            if it.get("has_session") and it.get("cookies"):
                r = await cl.add_account(it.get("label") or "ext_muse", it["cookies"])
                imported.append(r)
        return {"ok": len(imported) > 0, "found": len(items), "imported": imported}

    return {"ok": False, "error": f"Unsupported action={action!r} or missing account_id"}


@mcp.tool()
async def flowhs_check_us_proxy(proxy: str = "") -> dict[str, Any]:
    """Verify that a proxy (`ip:port:user:pass`) is working, resolves to a genuine United States (`US`) IP,
    and is NOT geo-blocked (`isGeoGated: false`) on `https://muse.ai/`."""
    _, _, default_proxy = _get_muse_settings()
    use_proxy = (proxy or default_proxy or "").strip()
    return await verify_us_proxy(use_proxy)


@mcp.tool()
async def flowhs_muse_auto_login(
    account_spec: str,
    proxy: str = "",
    headless: bool = False,
) -> dict[str, Any]:
    """Automatically log in one or more Muse (`muse.ai`) accounts in Camoufox using `email|inbox_url`
    (e.g. `joshuathompson796@evrnext.com|https://generator.email/inbox1/joshuathompson796@evrnext.com`),
    extract OTP code automatically from the inbox URL, capture `hatch_sess` + `hatch_gw` + `hatch_vml` cookies,
    and import the account(s) into `muse2api`.
    REQUIRES a valid US proxy (`countryCode == 'US'` and `isGeoGated == False`) — verifies the US proxy BEFORE opening Camoufox.

    Args:
        account_spec: One or more lines of `email|inbox_url` (or just `email` for generator.email).
        proxy: Required US residential proxy (`host:port:user:pass`). Falls back to `muse_proxy` in `config.json` if already configured.
        headless: Whether to run Camoufox headless (`False` by default).
    """
    base_url, api_key, default_proxy = _get_muse_settings()
    use_proxy = (proxy or default_proxy or "").strip()
    proxy_check = await verify_us_proxy(use_proxy)
    if not proxy_check.get("ok"):
        return {
            "ok": False,
            "error": proxy_check.get("error") or "Bắt buộc phải có Proxy US hợp lệ để đăng nhập tài khoản Muse!",
            "proxy_check": proxy_check,
        }

    # Save verified US proxy into config.json
    if proxy and proxy.strip():
        try:
            cfg_path = FLOWHS_DIR / "config.json"
            cfg = _load_flowhs_config()
            cfg["muse_proxy"] = use_proxy
            cfg["muse2api_proxy"] = use_proxy
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    lines = [ln.strip() for ln in (account_spec or "").splitlines() if ln.strip()]
    results = []
    for ln in lines:
        res = await auto_login_muse_account(
            email=ln,
            proxy_spec=use_proxy,
            headless=headless,
            base_url=base_url,
            api_key=api_key,
        )
        results.append(res)
    return {
        "ok": any(r.get("ok") for r in results),
        "total": len(results),
        "proxy_check": proxy_check,
        "results": results,
    }


@mcp.tool()
async def flowhs_clean_watermark(input_path: str, output_path: str = "") -> dict[str, Any]:
    """Remove watermarks from an image or video using PureBase reverse alpha unblending (`gwr.mjs`)."""
    src = Path(input_path)
    if not src.exists():
        return {"ok": False, "error": f"File not found: {input_path}"}
    dst = Path(output_path) if output_path else src
    if dst.resolve() != src.resolve():
        import shutil
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    cleaned = await _clean_media_watermark(dst)
    return {"ok": True, "path": cleaned}


def get_mcp_config_entry() -> dict[str, Any]:
    """Build the `mcpServers.flowhs` entry for the current installation (portable `.exe` or Python source)."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        mcp_exe = exe_dir / "Flowhs-MCP.exe"
        cmd_exe = mcp_exe if mcp_exe.exists() else Path(sys.executable).resolve()
        return {
            "command": str(cmd_exe),
            "args": ["--mcp"],
            "env": {"PYTHONIOENCODING": "utf-8"},
        }
    exe_dir = FLOWHS_DIR
    mcp_exe = exe_dir / "Flowhs-MCP.exe"
    if mcp_exe.exists():
        return {
            "command": str(mcp_exe),
            "args": ["--mcp"],
            "env": {"PYTHONIOENCODING": "utf-8"},
        }
    return {
        "command": "python",
        "args": [str(Path(__file__).resolve())],
        "env": {"PYTHONIOENCODING": "utf-8"},
    }


def register_mcp_to_local_agents() -> list[str]:
    """Automatically register `flowhs` MCP into installed AI agents on this PC
    (Antigravity, Claude Desktop, Claude Code, Cursor). Returns list of updated config paths."""
    entry = get_mcp_config_entry()
    home = Path.home()
    appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    candidates = [
        ("Antigravity", home / ".gemini" / "antigravity" / "mcp_config.json", True),
        ("Claude Desktop", appdata / "Claude" / "claude_desktop_config.json", False),
        ("Claude Code", home / ".claude.json", False),
        ("Cursor", home / ".cursor" / "mcp.json", False),
    ]
    updated: list[str] = []
    for label, cfg_path, create_parent in candidates:
        try:
            if not cfg_path.exists():
                if not create_parent and not cfg_path.parent.exists():
                    continue
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                data: dict[str, Any] = {}
            else:
                raw = cfg_path.read_text(encoding="utf-8").strip()
                data = json.loads(raw) if raw else {}
            if not isinstance(data, dict):
                continue
            servers = data.setdefault("mcpServers", {})
            changed = False
            if "veohs" in servers:
                servers.pop("veohs", None)
                changed = True
            if servers.get("flowhs") != entry:
                servers["flowhs"] = entry
                changed = True
            if changed:
                cfg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            updated.append(f"{label} ({cfg_path})")
        except Exception:
            pass
    return updated


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
