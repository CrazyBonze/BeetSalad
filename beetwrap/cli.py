import fcntl
import hashlib
import importlib.metadata
import itertools
import os
import sys
import threading
import urllib.request
from pathlib import Path

# Burnt orange-yellow ANSI color (256-color: color 214 ≈ orange-amber)
_COLOR = "\033[38;5;214m"
_RESET = "\033[0m"
_FRAMES = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

_GENRES_URL = "https://raw.githubusercontent.com/beetbox/beets/master/beetsplug/lastgenre/genres.txt"
_GENRES_TREE_URL = "https://raw.githubusercontent.com/beetbox/beets/master/beetsplug/lastgenre/genres-tree.yaml"


def _spinner(stop_event: threading.Event) -> None:
    msg = "Waiting for another beet instance to finish"
    while not stop_event.is_set():
        frame = next(_FRAMES)
        sys.stderr.write(f"\r{_COLOR}{frame} {msg}...{_RESET}")
        sys.stderr.flush()
        stop_event.wait(0.1)
    # Clear the spinner line
    sys.stderr.write(f"\r{' ' * (len(msg) + 6)}\r")
    sys.stderr.flush()


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=15) as resp:
        return resp.read().decode("utf-8")


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _stored_hash(hash_path: Path) -> str | None:
    if hash_path.exists():
        return hash_path.read_text(encoding="utf-8").strip()
    return None


def _custom_changed(custom_path: Path, hash_path: Path) -> bool:
    if not custom_path.exists():
        return False
    return _file_hash(custom_path) != _stored_hash(hash_path)


def _build_genre_file(
    url: str, output_path: Path, custom_path: Path, hash_path: Path, label: str
) -> None:
    sys.stderr.write(f"{_COLOR}⬇ Fetching {label} from beets repository...{_RESET}\n")
    sys.stderr.flush()
    try:
        content = _fetch(url)
        if custom_path.exists():
            custom = custom_path.read_text(encoding="utf-8")
            content = content.rstrip("\n") + "\n\n" + custom
            hash_path.write_text(_file_hash(custom_path), encoding="utf-8")
        output_path.write_text(content, encoding="utf-8")
        sys.stderr.write(f"{_COLOR}✓ {label} written to {output_path}{_RESET}\n")
    except Exception as exc:
        sys.stderr.write(f"{_COLOR}⚠ Could not fetch {label}: {exc}{_RESET}\n")
    sys.stderr.flush()


def _ensure_genre_files(beetsdir: Path) -> None:
    """Download upstream genre files if missing or if custom files have changed."""
    genres_path = beetsdir / "genres.txt"
    tree_path = beetsdir / "genres-tree.yaml"
    custom_genres = beetsdir / "genres.custom.txt"
    custom_tree = beetsdir / "genres-tree.custom.yaml"
    genres_hash = beetsdir / ".genres.custom.md5"
    tree_hash = beetsdir / ".genres-tree.custom.md5"

    if not genres_path.exists() or _custom_changed(custom_genres, genres_hash):
        _build_genre_file(
            _GENRES_URL, genres_path, custom_genres, genres_hash, "genres.txt"
        )

    if not tree_path.exists() or _custom_changed(custom_tree, tree_hash):
        _build_genre_file(
            _GENRES_TREE_URL, tree_path, custom_tree, tree_hash, "genres-tree.yaml"
        )


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] in ("-V", "--version"):
        version = importlib.metadata.version("beet")
        sys.stdout.write(f"{_COLOR}beet v{version}{_RESET}\n")
        return

    project_root = Path(__file__).resolve().parents[1]  # ../ (project root)
    beetsdir = project_root / "config"
    beetsdir.mkdir(parents=True, exist_ok=True)

    # Make the whole project self-contained: config + library db + state files.
    os.environ.setdefault("BEETSDIR", str(beetsdir))

    _ensure_genre_files(beetsdir)

    lock_file = beetsdir / "beet.lock"
    with open(lock_file, "w") as lock:
        # Try a non-blocking acquire first to see if we need to show the spinner
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Another instance holds the lock — show spinner and block
            stop_event = threading.Event()
            spinner_thread = threading.Thread(
                target=_spinner, args=(stop_event,), daemon=True
            )
            spinner_thread.start()
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
            finally:
                stop_event.set()
                spinner_thread.join()

        try:
            from beets.ui import main as beet_main

            beet_main()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
