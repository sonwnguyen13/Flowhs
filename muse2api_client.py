"""HTTP client & helpers for a muse2api gateway (https://github.com/crisng95/muse2api).

muse2api exposes the muse.ai web app as an OpenAI-compatible API:
    POST /v1/images/generations   prompt -> image(s)   (synchronous)
    POST /v1/videos               prompt (+first frame) -> task
    GET  /v1/videos/{task_id}     task status; result.url when succeeded
    GET  /v1/media/{name}         download generated media (public)
    GET  /readyz                  driver health + available accounts

Every `/v1/*` call carries `Authorization: Bearer <MUSE2API_KEY>`.
Input images for video generation (I2V) are sent inline as base64 data URLs.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import unquote, urlparse

import aiohttp
import certifi
import ssl

logger = logging.getLogger(__name__)

TERMINAL_OK = "succeeded"
TERMINAL_FAILED = ("failed", "cancelled")

DEFAULT_MUSE2API_URL = os.environ.get("MUSE2API_URL", "http://127.0.0.1:18610").strip().rstrip("/")
DEFAULT_MUSE2API_KEY = os.environ.get("MUSE2API_KEY", "muse2api-local").strip() or "muse2api-local"
DEFAULT_IMAGE_MODEL = os.environ.get("MUSE2API_IMAGE_MODEL", "muse-image").strip() or "muse-image"
DEFAULT_VIDEO_MODEL = os.environ.get("MUSE2API_VIDEO_MODEL", "muse-video").strip() or "muse-video"
DEFAULT_VIDEO_SECONDS = int(os.environ.get("MUSE2API_VIDEO_SECONDS", "8") or 8)

_MUSE2API_REPO_DIR = Path(__file__).resolve().parent.parent / "muse2api"
_SERVER_START_LOCK: Optional[asyncio.Lock] = None
_SERVER_PROC: Optional[subprocess.Popen] = None


def _resolve_default_api_key(explicit_key: str = "") -> str:
    key = (explicit_key or "").strip()
    if key:
        return key
    try:
        key_file = _MUSE2API_REPO_DIR / "data" / "api_key"
        if key_file.is_file():
            k = key_file.read_text(encoding="utf-8").strip()
            if k:
                return k
    except Exception:
        pass
    return DEFAULT_MUSE2API_KEY


async def ensure_local_muse2api_running(base_url: str = DEFAULT_MUSE2API_URL, *, proxy: str = "") -> bool:
    """If `base_url` points to localhost:18610 and isn't running yet, auto-start `muse2api`."""
    global _SERVER_START_LOCK, _SERVER_PROC
    parsed = urlparse(base_url)
    if parsed.hostname not in ("127.0.0.1", "localhost"):
        return False
    if not (_MUSE2API_REPO_DIR / "src" / "muse2api").is_dir():
        return False

    if _SERVER_START_LOCK is None:
        _SERVER_START_LOCK = asyncio.Lock()

    async with _SERVER_START_LOCK:
        # Quick health check
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1.5)) as s:
                async with s.get(f"{base_url.rstrip('/')}/healthz") as r:
                    if r.status == 200:
                        return True
        except Exception:
            pass

        logger.info("Auto-starting local muse2api gateway at %s ...", base_url)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_MUSE2API_REPO_DIR / "src") + os.pathsep + env.get("PYTHONPATH", "")
        if proxy:
            env["MUSE2API_BROWSER_PROXY"] = proxy
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            _SERVER_PROC = subprocess.Popen(
                [sys.executable, "-m", "muse2api"],
                cwd=str(_MUSE2API_REPO_DIR),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except Exception as e:
            logger.warning("Failed to spawn local muse2api server: %s", e)
            return False

        # Wait up to 8s for /healthz
        for _ in range(16):
            await asyncio.sleep(0.5)
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1.5)) as s:
                    async with s.get(f"{base_url.rstrip('/')}/healthz") as r:
                        if r.status == 200:
                            logger.info("Local muse2api gateway is up at %s", base_url)
                            return True
            except Exception:
                pass
        return False


class Muse2APIError(Exception):
    """A muse2api call failed. `code` is the gateway's error code when known."""

    def __init__(self, message: str, *, status: Optional[int] = None, code: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code

    def __str__(self) -> str:
        prefix = f"muse2api {self.code}" if self.code else "muse2api"
        if self.status:
            prefix += f" (HTTP {self.status})"
        return f"{prefix}: {self.message}"


def _file_url_to_path(url: str) -> Path:
    parsed = urlparse(url)
    raw_path = unquote(parsed.path or "")
    if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
        raw_path = raw_path[1:]
    return Path(raw_path)


def _normalize_size(aspect: Any) -> str:
    s = str(getattr(aspect, "value", aspect) or "").upper()
    if "PORTRAIT" in s or "9:16" in s or "VERTICAL" in s:
        return "9:16"
    if "SQUARE" in s or "1:1" in s:
        return "1:1"
    if "3:4" in s or "THREE_FOUR" in s:
        return "3:4"
    if "4:3" in s or "FOUR_THREE" in s:
        return "4:3"
    return "16:9"


def _ssl_connector() -> aiohttp.TCPConnector:
    try:
        ctx = ssl.create_default_context(cafile=certifi.where())
        return aiohttp.TCPConnector(ssl=ctx)
    except Exception:
        return aiohttp.TCPConnector(ssl=False)


async def to_data_url(ref: Union[str, Path], *, timeout: float = 60.0) -> str:
    """Turn a file:// path, local Path, http(s) URL or data URL into a base64 data URL."""
    ref_str = str(ref or "").strip()
    if not ref_str:
        raise ValueError("empty image reference")
    if ref_str.startswith("data:"):
        return ref_str
    parsed = urlparse(ref_str)
    if parsed.scheme in ("http", "https"):
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(connector=_ssl_connector(), timeout=timeout_cfg) as session:
            async with session.get(ref_str, allow_redirects=True) as resp:
                resp.raise_for_status()
                data = await resp.read()
                mime = (resp.headers.get("content-type") or "").split(";")[0].strip()
        mime = mime or mimetypes.guess_type(parsed.path)[0] or "image/png"
    else:
        path = _file_url_to_path(ref_str) if parsed.scheme == "file" else Path(ref_str)
        if not path or not path.is_file():
            raise FileNotFoundError(f"image not found: {ref_str}")
        data = await asyncio.to_thread(path.read_bytes)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def parse_muse_cookies(raw_text: str) -> dict[str, str]:
    """Parse muse.ai cookies (`hatch_sess`, `hatch_gw`, `hatch_vml`, `hatch_native_auth_device`)
    from Cookie-Editor JSON array, dict JSON, Netscape format, or `k=v; k2=v2` header string."""
    text = (raw_text or "").strip()
    if not text:
        return {}
    out: dict[str, str] = {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("cookies"), dict):
                return {str(k): str(v) for k, v in parsed["cookies"].items() if v}
            return {str(k): str(v) for k, v in parsed.items() if isinstance(v, (str, int, float))}
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and item.get("name") and item.get("value") is not None:
                    out[str(item["name"]).strip()] = str(item["value"]).strip()
            if out:
                return out
    except Exception:
        pass

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 7:
                out[parts[5].strip()] = parts[6].strip()
                continue
        for seg in line.split(";"):
            if "=" in seg:
                k, v = seg.split("=", 1)
                out[k.strip()] = v.strip()
    return out


class Muse2APIClient:
    """Async client for the muse2api media endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_MUSE2API_URL,
        api_key: str = "",
        *,
        timeout_s: float = 1800.0,
        poll_s: float = 5.0,
    ) -> None:
        self.base_url = (base_url or DEFAULT_MUSE2API_URL).strip().rstrip("/")
        self.api_key = _resolve_default_api_key(api_key)
        self.timeout_s = timeout_s
        self.poll_s = poll_s

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def _headers(self) -> dict[str, str]:
        hdrs = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/133.0.0.0 Safari/537.36",
        }
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        return hdrs

    def _url(self, path: str) -> str:
        return path if path.startswith(("http://", "https://")) else f"{self.base_url}{path}"

    async def _request(self, method: str, path: str, *, timeout: float = 60.0, json_body: Any = None) -> dict:
        url = self._url(path)
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        try:
            async with aiohttp.ClientSession(connector=_ssl_connector(), timeout=timeout_cfg) as session:
                async with session.request(method, url, headers=self._headers(), json=json_body) as resp:
                    text = await resp.text()
                    status = resp.status
                    reason = resp.reason or ""
        except aiohttp.ClientConnectorError as e:
            # Auto-start local muse2api server if pointing to localhost and retry once
            if await ensure_local_muse2api_running(self.base_url):
                try:
                    async with aiohttp.ClientSession(connector=_ssl_connector(), timeout=timeout_cfg) as session:
                        async with session.request(method, url, headers=self._headers(), json=json_body) as resp:
                            text = await resp.text()
                            status = resp.status
                            reason = resp.reason or ""
                except Exception as e2:
                    raise Muse2APIError(f"{method} {path} failed: {e2}", code="connection_error") from e2
            else:
                raise Muse2APIError(f"{method} {path} failed: {e}", code="connection_error") from e
        except asyncio.TimeoutError as e:
            raise Muse2APIError(f"{method} {path} timed out after {timeout:.0f}s", code="timeout") from e
        except aiohttp.ClientError as e:
            raise Muse2APIError(f"{method} {path} failed: {e}", code="connection_error") from e

        if status >= 400:
            try:
                body = json.loads(text)
            except Exception:
                body = None
            err = body.get("error") if isinstance(body, dict) else None
            if isinstance(err, dict):
                raise Muse2APIError(
                    err.get("message") or reason,
                    status=status,
                    code=err.get("code") or "",
                )
            detail = body.get("detail") if isinstance(body, dict) else None
            if isinstance(detail, str) and detail:
                raise Muse2APIError(detail, status=status)
            raise Muse2APIError(text[:300] or reason, status=status)

        try:
            return json.loads(text)
        except Exception as e:
            raise Muse2APIError(f"{method} {path} returned non-JSON", status=status) from e

    async def readyz(self) -> dict:
        """Check gateway driver health and available accounts (`GET /readyz`)."""
        return await self._request("GET", "/readyz", timeout=10.0)

    async def list_accounts(self) -> Any:
        """List accounts configured on the muse2api gateway (`GET /admin/accounts`)."""
        return await self._request("GET", "/admin/accounts", timeout=15.0)

    async def add_account(
        self,
        label: str,
        cookies: dict[str, str],
        cookie_expires: Optional[dict[str, int]] = None,
    ) -> dict:
        """Import a muse.ai account into the gateway (`POST /admin/accounts`)."""
        payload: dict[str, Any] = {"label": label or "acc1", "cookies": cookies}
        if cookie_expires:
            payload["cookie_expires"] = cookie_expires
        return await self._request("POST", "/admin/accounts", json_body=payload, timeout=20.0)

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_IMAGE_MODEL,
        size: Optional[str] = "16:9",
    ) -> bytes:
        """Generate one image and return its raw bytes."""
        body: dict[str, Any] = {
            "prompt": prompt,
            "model": model or DEFAULT_IMAGE_MODEL,
            "n": 1,
            "response_format": "b64_json",
        }
        if size:
            body["size"] = size
        data = await self._request("POST", "/v1/images/generations", json_body=body, timeout=self.timeout_s)
        items = data.get("data") or []
        if not items:
            raise Muse2APIError("image generation returned no data", code="empty_result")
        item = items[0]
        if item.get("b64_json"):
            return base64.b64decode(item["b64_json"])
        if item.get("url"):
            return await self.download(item["url"])
        raise Muse2APIError("image generation returned neither b64_json nor url", code="empty_result")

    async def create_video(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_VIDEO_MODEL,
        size: Optional[str] = "16:9",
        seconds: Optional[int] = DEFAULT_VIDEO_SECONDS,
        image: Optional[str] = None,
    ) -> dict:
        body: dict[str, Any] = {"prompt": prompt, "model": model or DEFAULT_VIDEO_MODEL}
        if size:
            body["size"] = size
        if seconds:
            body["seconds"] = int(seconds)
        if image:
            body["image"] = image
        return await self._request("POST", "/v1/videos", json_body=body, timeout=120.0)

    async def get_video(self, task_id: str) -> dict:
        return await self._request("GET", f"/v1/videos/{task_id}", timeout=30.0)

    async def wait_video(self, task_id: str) -> dict:
        """Poll a video task until it finishes. Returns the task's `result` dict."""
        deadline = time.monotonic() + self.timeout_s
        transient = 0
        while True:
            try:
                task = await self.get_video(task_id)
                transient = 0
            except Muse2APIError as e:
                if e.status == 404 or transient >= 3:
                    raise
                transient += 1
                logger.warning("muse2api: poll %s failed (%s), retrying", task_id, e)
                task = {}
            status = (task.get("status") or "").lower()
            if status == TERMINAL_OK:
                result = task.get("result") or {}
                if not result.get("url"):
                    raise Muse2APIError(f"video task {task_id} succeeded without a url", code="empty_result")
                return result
            if status in TERMINAL_FAILED:
                err = task.get("error")
                if isinstance(err, dict):
                    raise Muse2APIError(err.get("message") or f"video task {status}", code=err.get("code") or status)
                raise Muse2APIError(str(err or f"video task {status}"), code=status)
            if time.monotonic() >= deadline:
                raise Muse2APIError(
                    f"video task {task_id} timed out after {self.timeout_s:.0f}s (last status={status})",
                    code="timeout",
                )
            await asyncio.sleep(self.poll_s)

    async def download(self, url_or_path: str) -> bytes:
        url = self._url(url_or_path)
        timeout_cfg = aiohttp.ClientTimeout(total=120.0)
        try:
            async with aiohttp.ClientSession(connector=_ssl_connector(), timeout=timeout_cfg) as session:
                async with session.get(url, headers=self._headers(), allow_redirects=True) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        raise Muse2APIError(text[:300] or (resp.reason or "download failed"), status=resp.status)
                    return await resp.read()
        except aiohttp.ClientError as e:
            raise Muse2APIError(f"GET {url_or_path} failed: {e}", code="connection_error") from e


def get_muse_client(cfg: Optional[dict] = None) -> Muse2APIClient:
    """Create a Muse2APIClient from Flowhs config dict (or environment variables)."""
    if cfg is None:
        try:
            import llm
            cfg = llm.load_config()
        except Exception:
            cfg = {}
    base_url = (cfg.get("muse2api_url") or os.environ.get("MUSE2API_URL") or DEFAULT_MUSE2API_URL).strip()
    api_key = _resolve_default_api_key(cfg.get("muse2api_key") or os.environ.get("MUSE2API_KEY") or "")
    return Muse2APIClient(base_url=base_url, api_key=api_key)


async def generate_image_muse(
    prompt: str,
    aspect: Any = "16:9",
    out_path: Optional[Union[str, Path]] = None,
    cfg: Optional[dict] = None,
) -> dict[str, str]:
    """High-level helper for Flowhs: generate an image via muse2api, optionally write to `out_path`."""
    client = get_muse_client(cfg)
    size = _normalize_size(aspect)
    model = ((cfg or {}).get("muse2api_image_model") or DEFAULT_IMAGE_MODEL).strip() or "muse-image"
    img_bytes = await client.generate_image(prompt, model=model, size=size)
    mid = str(uuid.uuid4())
    if out_path:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(dest.write_bytes, img_bytes)
        url = dest.resolve().as_uri()
    else:
        b64 = base64.b64encode(img_bytes).decode()
        url = f"data:image/png;base64,{b64}"
    return {"url": url, "generation_id": mid, "media_id": mid, "status": "done"}


async def generate_video_muse(
    image_source: Union[str, Path],
    prompt: str,
    aspect: Any = "16:9",
    out_path: Optional[Union[str, Path]] = None,
    cfg: Optional[dict] = None,
    seconds: Optional[int] = None,
) -> dict[str, str]:
    """High-level helper for Flowhs: generate I2V video via muse2api from `image_source`."""
    client = get_muse_client(cfg)
    size = _normalize_size(aspect)
    model = ((cfg or {}).get("muse2api_video_model") or DEFAULT_VIDEO_MODEL).strip() or "muse-video"
    dur = seconds or int((cfg or {}).get("muse2api_video_seconds") or DEFAULT_VIDEO_SECONDS)
    data_url = await to_data_url(image_source)
    task = await client.create_video(prompt, model=model, size=size, seconds=dur, image=data_url)
    task_id = task.get("id")
    if not task_id:
        raise Muse2APIError("muse2api: video task created without an id", code="empty_result")
    result = await client.wait_video(task_id)
    vid_bytes = await client.download(result["url"])
    mid = str(uuid.uuid4())
    if out_path:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(dest.write_bytes, vid_bytes)
        url = dest.resolve().as_uri()
    else:
        url = client._url(result["url"])
    return {"url": url, "generation_id": mid, "media_id": mid, "status": "done"}


_US_BROWSER_BRIDGE_PORT: Optional[int] = None
_US_BROWSER_BRIDGE_SPEC: Optional[str] = None


def update_muse2api_env_proxy(proxy_spec: str) -> None:
    """Save `MUSE2API_BROWSER_PROXY` to `muse2api/.env`."""
    try:
        env_file = _MUSE2API_REPO_DIR / ".env"
        if not env_file.parent.is_dir():
            return
        lines = []
        if env_file.is_file():
            lines = [
                l for l in env_file.read_text(encoding="utf-8").splitlines()
                if not l.startswith("MUSE2API_BROWSER_PROXY=")
            ]
        lines.append(f"MUSE2API_BROWSER_PROXY={proxy_spec.strip()}")
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to update muse2api .env proxy: %s", e)


def launch_muse_us_browser(proxy_spec: str = "", on_imported: Optional[Any] = None) -> str:
    """Launch Camoufox anti-detect browser routed through `proxy_spec` (`ip:port:user:pass`)
    with US timezone (`America/Los_Angeles`), `en-US` locale, and WebRTC leak protection.
    Automatically watches `https://muse.ai` cookies and imports them into `muse2api` as soon as login completes."""
    import threading

    spec = (proxy_spec or "").strip()
    if spec:
        update_muse2api_env_proxy(spec)

    up_host, up_port, p_user, p_pass = "", 0, "", ""
    if spec:
        if "://" not in spec:
            parts = spec.split(":")
            if len(parts) == 4:
                up_host, up_port, p_user, p_pass = parts[0], int(parts[1]), parts[2], parts[3]
            elif len(parts) == 2:
                up_host, up_port = parts[0], int(parts[1])
        else:
            p = urlparse(spec)
            up_host = p.hostname or ""
            up_port = int(p.port or 80)
            p_user = p.username or ""
            p_pass = p.password or ""

    proxy_dict = None
    if up_host and up_port:
        proxy_dict = {"server": f"http://{up_host}:{up_port}"}
        if p_user:
            proxy_dict["username"] = p_user
            proxy_dict["password"] = p_pass

    def _run_camoufox():
        async def _worker():
            from playwright.async_api import async_playwright
            from camoufox.async_api import AsyncNewBrowser
            from browser_engine import BrowserEngine

            flowhs_dir = Path(__file__).resolve().parent
            profile_dir = (flowhs_dir / "data" / "muse_us_camoufox_profile").resolve()
            profile_dir.mkdir(parents=True, exist_ok=True)
            lock_file = profile_dir / "parent.lock"
            if lock_file.exists():
                try:
                    lock_file.unlink()
                except Exception:
                    pass

            engine = BrowserEngine(headless=False)
            bundled_exe = engine._get_camoufox_executable()
            extra_kwargs = {}
            if bundled_exe:
                extra_kwargs["executable_path"] = str(bundled_exe)

            async with async_playwright() as pw:
                ctx = await AsyncNewBrowser(
                    pw,
                    headless=False,
                    persistent_context=True,
                    user_data_dir=str(profile_dir),
                    window=(1280, 720),
                    viewport={"width": 1280, "height": 720},
                    locale="en-US",
                    timezone_id="America/Los_Angeles",
                    block_webrtc=True,
                    proxy=proxy_dict,
                    firefox_user_prefs={"security.webauth.webauthn": False},
                    **extra_kwargs,
                )
                page = ctx.pages[0] if ctx.pages else await ctx.new_page()
                try:
                    await page.goto("https://muse.ai/", wait_until="domcontentloaded", timeout=45000)
                except Exception:
                    pass

                imported_once = False
                while not page.is_closed():
                    await asyncio.sleep(2.0)
                    try:
                        all_cks = await ctx.cookies(["https://muse.ai", "https://www.muse.ai"])
                        ck_map = {}
                        exp_map = {}
                        for c in all_cks:
                            name = c.get("name")
                            val = c.get("value")
                            if name and val:
                                ck_map[name] = val
                                if c.get("expires") and c["expires"] > 0:
                                    exp_map[name] = int(c["expires"])
                        if ck_map.get("hatch_sess") and ck_map.get("hatch_gw") and not imported_once:
                            await asyncio.sleep(3.0)
                            all_cks = await ctx.cookies(["https://muse.ai", "https://www.muse.ai"])
                            for c in all_cks:
                                if c.get("name") and c.get("value"):
                                    ck_map[c["name"]] = c["value"]
                                    if c.get("expires") and c["expires"] > 0:
                                        exp_map[c["name"]] = int(c["expires"])
                            await ensure_local_muse2api_running()
                            cl = Muse2APIClient()
                            res = await cl.add_account("Muse-Camoufox-US", ck_map, exp_map)
                            imported_once = True
                            logger.info("Auto-imported Muse US cookies from Camoufox: %s", res)
                            if callable(on_imported):
                                try:
                                    on_imported(res)
                                except Exception:
                                    pass
                    except Exception:
                        if page.is_closed():
                            break
                try:
                    await ctx.close()
                except Exception:
                    pass

        try:
            asyncio.run(_worker())
        except Exception as e:
            logger.warning("Camoufox Muse US session ended: %s", e)

    threading.Thread(target=_run_camoufox, daemon=True).start()
    return f"http://{up_host}:{up_port}" if up_host else "direct"


async def auto_login_muse_account(
    email: str,
    inbox_url: str = "",
    *,
    proxy_spec: str = "",
    headless: bool = False,
    base_url: str = DEFAULT_MUSE2API_URL,
    api_key: str = "",
) -> dict[str, Any]:
    """Automatically log in or sign up a Muse (`muse.ai`) account in Camoufox using `email|inbox_url`
    (e.g. `generator.email` or custom webmail inbox), extract all cookies (`hatch_sess`, `hatch_gw`, `hatch_vml`),
    and import the account into `muse2api`."""
    import re
    from camoufox.async_api import AsyncNewBrowser
    from browser_engine import BrowserEngine

    email = (email or "").strip()
    inbox_url = (inbox_url or "").strip()
    if "|" in email and not inbox_url:
        parts_e = email.split("|", 1)
        email, inbox_url = parts_e[0].strip(), parts_e[1].strip()
    if not inbox_url and "@" in email:
        inbox_url = f"https://generator.email/inbox1/{email}"

    spec = (proxy_spec or "").strip()
    if spec:
        update_muse2api_env_proxy(spec)

    up_host, up_port, p_user, p_pass = "", 0, "", ""
    if spec:
        if "://" not in spec:
            parts = spec.split(":")
            if len(parts) == 4:
                up_host, up_port, p_user, p_pass = parts[0], int(parts[1]), parts[2], parts[3]
            elif len(parts) == 2:
                up_host, up_port = parts[0], int(parts[1])
        else:
            p = urlparse(spec)
            up_host = p.hostname or ""
            up_port = int(p.port or 80)
            p_user = p.username or ""
            p_pass = p.password or ""

    proxy_dict = None
    if up_host and up_port:
        proxy_dict = {"server": f"http://{up_host}:{up_port}"}
        if p_user:
            proxy_dict["username"] = p_user
            proxy_dict["password"] = p_pass

    engine = BrowserEngine(headless=headless)
    await engine.start()
    profile_path = engine.get_user_data_dir(email)
    lock_file = profile_path / "parent.lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
        except Exception:
            pass

    bundled_exe = engine._get_camoufox_executable()
    extra_kwargs = {"executable_path": str(bundled_exe)} if bundled_exe else {}

    ctx = await AsyncNewBrowser(
        engine.pw,
        headless=headless,
        persistent_context=True,
        user_data_dir=str(profile_path),
        window=(1280, 720),
        viewport={"width": 1280, "height": 720},
        locale="en-US",
        timezone_id="America/New_York",
        block_webrtc=True,
        proxy=proxy_dict,
        firefox_user_prefs={"security.webauth.webauthn": False},
        **extra_kwargs,
    )
    try:
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto("https://muse.ai/", wait_until="domcontentloaded", timeout=45000)
        await asyncio.sleep(3.5)

        body_text = await page.evaluate("() => document.body ? document.body.innerText : ''")
        if "Mobile number or email" in body_text or "Log in" in body_text:
            inp = page.locator("input").first
            if await inp.count() > 0:
                if not await inp.is_visible():
                    login_btn = page.locator("//button[contains(., 'Log in')]").first
                    if await login_btn.count() > 0:
                        await login_btn.click()
                        await asyncio.sleep(2.0)
                if await inp.is_visible():
                    await inp.fill(email)
                    await asyncio.sleep(1.0)
                    cont_btn = page.locator("//button[contains(., 'Continue') and not(@disabled)]").first
                    if await cont_btn.count() > 0 and await cont_btn.is_visible():
                        await cont_btn.click()
                        await asyncio.sleep(4.5)

        body_text = await page.evaluate("() => document.body ? document.body.innerText : ''")
        if "Enter your code" in body_text or "6-digit code" in body_text or "code to log in" in body_text:
            epage = await ctx.new_page()
            otp_code = ""
            try:
                if "generator.email" in inbox_url and "@" in email:
                    user_part, dom_part = email.split("@", 1)
                    await ctx.add_cookies([
                        {
                            "name": "surl",
                            "value": f"{dom_part}/{user_part}",
                            "domain": "generator.email",
                            "path": "/",
                        }
                    ])
                await epage.goto(inbox_url, wait_until="domcontentloaded", timeout=35000)
                for _ in range(18):
                    await asyncio.sleep(3.5)
                    btxt = await epage.evaluate("() => document.body ? document.body.innerText : ''")
                    if "No restrictions" in btxt:
                        clink = epage.locator("//a[contains(., 'Continue')]").first
                        if await clink.count() > 0:
                            await clink.click()
                            await asyncio.sleep(2.5)
                            continue
                    m = re.search(
                        r"(?:Confirmation code|security code|code is|confirm your account|Meta)[\s\S]{0,120}?\b(\d{6})\b",
                        btxt,
                        re.I,
                    )
                    if m:
                        otp_code = m.group(1)
                        break
                    for line in btxt.splitlines():
                        if re.fullmatch(r"\d{6}", line.strip()):
                            otp_code = line.strip()
                            break
                    if otp_code:
                        break
                    try:
                        await epage.reload(wait_until="domcontentloaded", timeout=20000)
                    except Exception:
                        pass
            finally:
                try:
                    await epage.close()
                except Exception:
                    pass

            if otp_code:
                await page.bring_to_front()
                await asyncio.sleep(0.8)
                inputs = page.locator("input:not([type='checkbox'])")
                icnt = await inputs.count()
                if icnt == 1:
                    await inputs.first.fill(otp_code)
                elif icnt >= 6:
                    for i in range(6):
                        await inputs.nth(i).fill(otp_code[i])
                else:
                    await page.keyboard.type(otp_code, delay=80)
                await asyncio.sleep(3.0)
                conf_btn = page.locator("//button[not(@disabled) and (contains(., 'Confirm') or contains(., 'Continue'))]").first
                if await conf_btn.count() > 0 and await conf_btn.is_visible():
                    try:
                        await conf_btn.click(timeout=4000)
                    except Exception:
                        pass

        # Wait up to 45s for onboarding or main app cookies
        for _ in range(18):
            await asyncio.sleep(2.5)
            btxt = await page.evaluate("() => document.body ? document.body.innerText.slice(0, 600) : ''")
            low_b = btxt.lower()
            if ("birthday" in low_b or "first name" in low_b) and "creating account" not in low_b:
                try:
                    fn_inp = page.locator("input[placeholder='First name'], input[aria-label='First name']").first
                    if await fn_inp.count() > 0 and await fn_inp.is_visible():
                        await fn_inp.fill("Alex", timeout=3000)
                    ln_inp = page.locator("input[placeholder='Last name'], input[aria-label='Last name']").first
                    if await ln_inp.count() > 0 and await ln_inp.is_visible():
                        await ln_inp.fill("Smith", timeout=3000)
                    yr_btn = page.locator("button[aria-label='Year']").first
                    if await yr_btn.count() > 0 and await yr_btn.is_visible():
                        if "1992" not in (await yr_btn.inner_text() or ""):
                            await yr_btn.click(timeout=3000)
                            await asyncio.sleep(0.8)
                            opt92 = page.locator("//*[@role='option' and contains(., '1992')]").first
                            if await opt92.count() > 0:
                                await opt92.scroll_into_view_if_needed()
                                await opt92.click(timeout=3000)
                    b_btn = page.locator("button[data-pel-click='login_birthday_continue_click']").first
                    if await b_btn.count() > 0 and (await b_btn.get_attribute("disabled")) is None:
                        await b_btn.click(timeout=4000)
                        await asyncio.sleep(4.0)
                        continue
                except Exception:
                    pass

            all_cks = await ctx.cookies(["https://muse.ai", "https://www.muse.ai"])
            ck_map = {c["name"]: c["value"] for c in all_cks if c.get("name") and c.get("value")}
            exp_map = {c["name"]: int(c["expires"]) for c in all_cks if c.get("name") and c.get("expires") and c["expires"] > 0}
            if ck_map.get("hatch_sess") and ck_map.get("hatch_gw"):
                await asyncio.sleep(2.0)
                all_cks = await ctx.cookies(["https://muse.ai", "https://www.muse.ai"])
                ck_map = {c["name"]: c["value"] for c in all_cks if c.get("name") and c.get("value")}
                exp_map = {c["name"]: int(c["expires"]) for c in all_cks if c.get("name") and c.get("expires") and c["expires"] > 0}
                await ensure_local_muse2api_running(base_url, proxy=spec)
                cl = Muse2APIClient(base_url, api_key)
                res = await cl.add_account(email, ck_map, exp_map)
                return {"ok": True, "email": email, "imported": res}

        all_cks = await ctx.cookies(["https://muse.ai", "https://www.muse.ai"])
        return {
            "ok": False,
            "email": email,
            "url": page.url,
            "cookies_found": [c.get("name") for c in all_cks],
            "error": "Did not acquire both hatch_sess and hatch_gw within timeout",
        }
    finally:
        try:
            await ctx.close()
        except Exception:
            pass
        try:
            await engine.stop()
        except Exception:
            pass
