from __future__ import annotations

import argparse
import sys
from pathlib import Path

import webview as pywebview

if __package__ in {None, ""}:
    # Ensure project root is on sys.path when launched as `python webview_UI/app.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from webview_UI.api import WebviewApi  # type: ignore[import-self]
else:
    from .api import WebviewApi

def resolve_assets_dir() -> Path:
    """Locate the UI assets, handling both local and PyInstaller contexts."""

    frozen_base = getattr(sys, "_MEIPASS", None)
    if frozen_base:
        candidate = Path(frozen_base) / "webview_assets"
        if candidate.exists():
            return candidate

    return Path(__file__).resolve().parent / "assets"


ASSETS_DIR = resolve_assets_dir()
INDEX_FILE = ASSETS_DIR / "index.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the PyWebview Echo Policy Calculator UI.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable PyWebview debug mode (opens developer tools where supported).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Unable to locate UI assets at {INDEX_FILE!s}")

    api = WebviewApi()
    pywebview.create_window(
        "声骸强化策略计算器",
        str(INDEX_FILE),
        js_api=api,
        width=1100,
        height=900,
        min_size=(1100, 500),
        background_color="#f8fafc",
    )

    pywebview.start(debug=args.debug, http_server=True)


if __name__ == "__main__":
    main()
