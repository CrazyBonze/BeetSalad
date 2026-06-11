"""
beets plugin: desktop notification when beets needs user input.

On Xubuntu/XFCE (or any desktop with notify-send), fires a toast whenever
beets blocks for interactive input — album matches, duplicate resolution,
confirmation prompts, anything.

How it works:
  beets.ui.input_options() is the single chokepoint where beets presents
  interactive prompts and waits for a keypress.  This plugin monkey-patches
  that function to fire a notify-send right before beets blocks for input.

  This is more universal than patching TerminalImportSession.choose_match,
  because it catches every interactive prompt in beets, not just album/track
  matching.

Installation:
  1. Drop this file into your beets plugin directory, e.g.:
       ~/.config/beets/beetsplug/importnotify.py
  2. Make sure that directory is in your beets pluginpath:
       pluginpath:
         - ~/.config/beets/beetsplug
  3. Enable the plugin:
       plugins: importnotify
  4. Optionally configure:
       importnotify:
         urgency: normal   # low | normal | critical
         icon: dialog-question

Usage:
  Just `beet import /path` as usual. Walk away. Whenever beets pauses for
  input you'll get a desktop notification.
"""

from __future__ import annotations

import subprocess

import beets.ui
import beets.ui.commands
from beets.plugins import BeetsPlugin


def _notify(
    summary: str,
    body: str = "",
    urgency: str = "normal",
    icon: str = "dialog-question",
) -> None:
    """Fire-and-forget desktop notification via notify-send."""
    cmd = [
        "notify-send",
        "--app-name=beets",
        f"--urgency={urgency}",
        f"--icon={icon}",
        summary,
    ]
    if body:
        cmd.append(body)
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        pass  # notify-send not installed; silent no-op


class ImportNotifyPlugin(BeetsPlugin):
    def __init__(self) -> None:
        super().__init__()
        self.config.add(
            {
                "urgency": "normal",  # notify-send urgency: low, normal, critical
                "icon": "dialog-question",  # freedesktop icon name
            }
        )
        self._patch_input_options()

    def _patch_input_options(self) -> None:
        urgency = self.config["urgency"].get(str)
        icon = self.config["icon"].get(str)
        log = self._log

        _orig_input_options = beets.ui.input_options

        def _input_options_wrapper(*args, **kwargs):
            options = args[0] if args else kwargs.get("options", [])

            # Build a short summary from the option names.
            if options:
                opts_str = ", ".join(
                    o.long if hasattr(o, "long") else str(o) for o in options
                )
                body = f"Options: {opts_str}"
            else:
                body = ""

            _notify("beets: input needed", body, urgency, icon)
            log.debug("Sent desktop notification (input_options)")

            return _orig_input_options(*args, **kwargs)

        # Patch the module-level function.
        beets.ui.input_options = _input_options_wrapper

        # Also patch in beets.ui.commands, which may have bound
        # input_options as a local name via `from beets.ui import ...`
        # before our plugin loaded.
        if hasattr(beets.ui.commands, "input_options"):
            beets.ui.commands.input_options = _input_options_wrapper

        log.debug("input_options notification hook installed.")
