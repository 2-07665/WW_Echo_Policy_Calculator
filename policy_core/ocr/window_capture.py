from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Tuple

from PIL import Image

if sys.platform != "win32":  # pragma: no cover - Windows only module
    raise ImportError("window_capture is only available on Windows.")

import mss  # type: ignore  # noqa: E402
import win32con  # type: ignore  # noqa: E402
import win32gui  # type: ignore  # noqa: E402

CropRegion = tuple[float, float, float, float]


class WindowNotFoundError(RuntimeError):
    """Raised when the requested window title cannot be resolved."""


@dataclass
class WindowCaptureConfig:
    title: str
    restore_window: bool = True


def _apply_crop(bbox: tuple[int, int, int, int], crop: CropRegion | None) -> tuple[int, int, int, int]:
    if crop is None:
        return bbox
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top

    crop_left, crop_top, crop_right, crop_bottom = crop
    new_left = left + math.floor(width * crop_left)
    new_top = top + math.floor(height * crop_top)
    new_right = left + math.ceil(width * crop_right)
    new_bottom = top + math.ceil(height * crop_bottom)
    return new_left, new_top, new_right, new_bottom


def _find_window(title: str) -> int:
    hwnd = win32gui.FindWindow(None, title)
    if hwnd:
        return hwnd
    raise WindowNotFoundError(f"No window found with exact title '{title}'.")


def _ensure_window_visible(hwnd: int) -> None:
    if not win32gui.IsWindow(hwnd):
        raise WindowNotFoundError("Window handle is no longer valid.")
    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        time.sleep(0.3)
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:  # noqa: BLE001 - best effort
        pass


def _window_bbox(hwnd: int) -> tuple[int, int, int, int]:
    client = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (client[0], client[1]))
    right, bottom = win32gui.ClientToScreen(hwnd, (client[2], client[3]))
    if left == right or top == bottom:
        raise RuntimeError("Window client area is zero-sized.")
    return left, top, right, bottom


class WindowCapture:
    """Capture RGB screenshots from a specific window title."""

    def __init__(self, config: WindowCaptureConfig):
        self.config = config

    def window_exists(self) -> bool:
        try:
            _find_window(self.config.title)
            return True
        except WindowNotFoundError:
            return False

    def grab(self, *, crop: CropRegion | None = None, restore: bool | None = None) -> Image.Image:
        hwnd = _find_window(self.config.title)
        if restore if restore is not None else self.config.restore_window:
            _ensure_window_visible(hwnd)
        bbox = _window_bbox(hwnd)
        left, top, right, bottom = _apply_crop(bbox, crop)
        with mss.mss() as capturer:
            shot = capturer.grab(
                {"left": left, "top": top, "width": right - left, "height": bottom - top}
            )
        return Image.frombytes("RGB", shot.size, shot.rgb)
