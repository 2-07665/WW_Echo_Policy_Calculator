from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import sys

from policy_core.ui_service import PolicyUIService


def resource_path(filename: str) -> Path:
    """Return the absolute path to a bundled resource, working with PyInstaller."""

    base_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []

    if getattr(sys, "frozen", False):
        exe_path = Path(sys.executable).resolve()
        exe_dir = exe_path.parent

        mac_app_root = next((parent for parent in exe_path.parents if parent.suffix == ".app"), None)
        if mac_app_root is not None:
            candidates.append(mac_app_root.parent / filename)  # alongside the .app bundle
            candidates.append(mac_app_root / filename)  # inside the .app root
            candidates.append(mac_app_root / "Contents" / "Resources" / filename)

        candidates.append(exe_dir / filename)  # alongside the executable itself
        candidates.append(exe_dir / "webview_UI" / filename)
        meipass = Path(getattr(sys, "_MEIPASS", exe_dir))
        candidates.append(meipass / filename)

    candidates.append(Path.cwd() / filename)
    candidates.append(base_dir / filename)

    ordered_candidates: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        ordered_candidates.append(path)

    for path in ordered_candidates:
        if path.exists():
            return path
    return ordered_candidates[0]


class WebviewApi:
    """Bridge exposed to the JavaScript layer within the PyWebview UI."""

    def __init__(self) -> None:
        preset_path = resource_path("character_preset.json")
        user_counts_path = resource_path("user_counts_data.json")
        self._service = PolicyUIService(
            preset_path=preset_path,
            user_counts_path=user_counts_path,
        )

    # ---- Policy operations -------------------------------------------------

    def bootstrap(self) -> dict[str, Any]:
        return self._service.bootstrap()

    def set_exp_refund_ratio(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._service.set_exp_refund_ratio(payload)

    def compute_policy(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._service.compute_policy(payload)

    def policy_suggestion(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._service.policy_suggestion(payload)

    # ---- Platform info -----------------------------------------------------

    def get_platform(self) -> str:
        return self._service.get_platform()
