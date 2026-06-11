"""
Navidrome sync plugin for beets (simple + gentle)

- Coalesce changes into at most ONE scan per beet run (at cli_exit).
- Skip scan on tag-only changes when watcher_mode is True.
- Always call startScan(fullScan=false) (incremental).
- Skip starting scan if Navidrome is already scanning (optional).
"""

from __future__ import annotations

import hashlib
import random
import string
from binascii import hexlify
from typing import Any, Dict, Optional, Tuple

import requests

from beets.plugins import BeetsPlugin


class NavidromeSync(BeetsPlugin):
    def __init__(self) -> None:
        super().__init__()

        self.config.add(
            {
                "enabled": True,
                "dry_run": False,
                "url": "http://localhost:4533",
                "user": "admin",
                "pass": "admin",
                "auth": "token",  # "token" or "password"
                "api_version": "1.16.1",
                "watcher_mode": True,
                "skip_if_scanning": True,
            }
        )

        self.config["user"].redact = True
        self.config["pass"].redact = True

        # Per-run flags
        self._saw_tag_write = False
        self._saw_real_move = False
        self._saw_remove = False

        # Hooks
        self.register_listener("after_write", self.on_after_write)
        self.register_listener("item_moved", self.on_item_moved)
        self.register_listener("item_removed", self.on_item_removed)
        self.register_listener("album_removed", self.on_album_removed)
        self.register_listener("cli_exit", self.on_cli_exit)

    # ---------------------------
    # Beets event handlers
    # ---------------------------

    def on_after_write(self, item, **kwargs) -> None:
        if not self.config["enabled"].get(bool):
            return
        self._saw_tag_write = True

    def on_item_moved(self, item, source, destination, **kwargs) -> None:
        if not self.config["enabled"].get(bool):
            return
        if source == destination:
            return
        self._saw_real_move = True

    def on_item_removed(self, item, **kwargs) -> None:
        if not self.config["enabled"].get(bool):
            return
        self._saw_remove = True

    def on_album_removed(self, lib=None, album=None, **kwargs) -> None:
        # Accept kwargs to be robust against beets calling conventions.
        if not self.config["enabled"].get(bool):
            return
        self._saw_remove = True

    def on_cli_exit(self, lib, **kwargs) -> None:
        if not self.config["enabled"].get(bool):
            return

        # Decide if scan is needed
        need_scan = False
        if self._saw_remove or self._saw_real_move:
            need_scan = True
        elif self._saw_tag_write and not self.config["watcher_mode"].get(bool):
            need_scan = True

        if not need_scan:
            return

        if self.config["skip_if_scanning"].get(bool):
            try:
                if self._is_scanning():
                    self._log.debug("NavidromeSync: already scanning; skipping startScan")
                    return
            except Exception as exc:
                # If status check fails, still attempt scan (best effort).
                self._log.debug("NavidromeSync: getScanStatus failed: {}", exc)

        self._start_scan()

    # ---------------------------
    # Subsonic API helpers
    # ---------------------------

    def _format_url(self, endpoint: str) -> str:
        base = self.config["url"].as_str().rstrip("/")
        return f"{base}/rest/{endpoint}"

    def _create_token(self) -> Tuple[str, str]:
        password = self.config["pass"].as_str()
        chars = string.ascii_letters + string.digits
        salt = "".join(random.choice(chars) for _ in range(6))
        token = hashlib.md5(f"{password}{salt}".encode("utf-8")).hexdigest()
        return salt, token

    def _base_payload(self) -> Dict[str, Any]:
        user = self.config["user"].as_str()
        auth = self.config["auth"].as_str()
        v = self.config["api_version"].as_str()

        payload: Dict[str, Any] = {"u": user, "v": v, "c": "beets", "f": "json"}

        if auth == "token":
            salt, token = self._create_token()
            payload.update({"t": token, "s": salt})
        elif auth == "password":
            password = self.config["pass"].as_str()
            encpass = hexlify(password.encode()).decode()
            payload.update({"p": f"enc:{encpass}"})
        else:
            raise RuntimeError(f"NavidromeSync: unknown auth mode: {auth}")

        return payload

    def _get(self, endpoint: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self._format_url(endpoint)
        params = self._base_payload()
        if extra:
            params.update(
                {
                    k: ("true" if v else "false") if isinstance(v, bool) else v
                    for k, v in extra.items()
                }
            )

        if self.config["dry_run"].get(bool):
            self._log.info("NavidromeSync dry_run: GET {} params={}", url, params)
            if endpoint == "getScanStatus":
                return {"subsonic-response": {"status": "ok", "scanStatus": {"scanning": False}}}
            return {"subsonic-response": {"status": "ok"}}

        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {data}")
        return data

    # ---------------------------
    # Scan ops
    # ---------------------------

    def _is_scanning(self) -> bool:
        data = self._get("getScanStatus")
        sr = data.get("subsonic-response", {})
        if sr.get("status") != "ok":
            raise RuntimeError(f"getScanStatus returned {data}")
        scan = sr.get("scanStatus") or {}
        return bool(scan.get("scanning", False))

    def _start_scan(self) -> None:
        try:
            self._get("startScan", {"fullScan": False})
            self._log.info("NavidromeSync: startScan triggered (fullScan=false)")
        except Exception as exc:
            self._log.error("NavidromeSync: startScan failed: {}", exc)