"""
Utilities for capturing frames of a window on Windows.

* `capture_frames(...)` yields `PIL.Image` frames for in-memory processing.
* `save_frames(...)` writes captures to disk.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path
from typing import Generator

import mss
from PIL import Image
import win32con
import win32gui

CropRegion = tuple[float, float, float, float]


def _apply_crop(bbox: tuple[int, int, int, int], crop: CropRegion) -> tuple[int, int, int, int]:
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
    """Return a window handle that matches the requested title."""
    hwnd = win32gui.FindWindow(None, title)
    if hwnd:
        return hwnd
    raise RuntimeError(f"No window found with exact title '{title}'.")


def _ensure_window_visible(hwnd: int) -> None:
    """Make sure the window is restored and foregrounded."""
    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        time.sleep(0.5)
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass


def _window_bbox(hwnd: int) -> tuple[int, int, int, int]:
    """Return window client bounds in screen coordinates."""
    client = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (client[0], client[1]))
    right, bottom = win32gui.ClientToScreen(hwnd, (client[2], client[3]))
    if left == right or top == bottom:
        raise RuntimeError("Window client area is zero-sized.")
    return left, top, right, bottom


def _grab_image(
    capturer: mss.mss,
    bbox: tuple[int, int, int, int],
    crop: CropRegion | None = None,
) -> Image.Image:
    if crop is not None:
        left, top, right, bottom = _apply_crop(bbox, crop)
    else:
        left, top, right, bottom = bbox
    shot = capturer.grab(
        {"left": left, "top": top, "width": right - left, "height": bottom - top}
    )
    return Image.frombytes("RGB", shot.size, shot.rgb)


def capture_frames(
    title: str,
    *,
    exact: bool = True,
    interval: float = 1.0,
    count: int = 0,
    crop: CropRegion | None = None,
) -> Generator[Image.Image, None, None]:
    """
    Yield frames from the specified window.

    Args:
        title: Window title to match exactly.
        exact: Require an exact title match; partial matching is no longer supported.
        interval: Seconds between captures.
        count: Number of frames to yield (0 = infinite).
        crop: Optional (left, top, right, bottom) percentages specifying a sub-region.
    """
    if not exact:
        raise ValueError("Partial title matching has been removed; set exact=True.")
    hwnd = _find_window(title)
    _ensure_window_visible(hwnd)
    sleep_duration = max(interval, 0.01)

    with mss.mss() as capturer:
        produced = 0
        while not count or produced < count:
            bbox = _window_bbox(hwnd)
            frame = _grab_image(capturer, bbox, crop=crop)
            produced += 1
            yield frame
            time.sleep(sleep_duration)


def save_frames(
    title: str,
    *,
    exact: bool = True,
    interval: float = 1.0,
    count: int = 0,
    output: Path | None = None,
    timestamped: bool = True,
    crop: CropRegion | None = None,
) -> None:
    """
    Save captured frames to disk.

    Args mirror `capture_frames`, plus:
        output: Target directory (defaults to ./captures).
        timestamped: If True, filenames use timestamps; otherwise sequential numbering.
    """
    if not exact:
        raise ValueError("Partial title matching has been removed; set exact=True.")

    destination = output if output is not None else Path("captures")
    destination.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(
        capture_frames(
            title,
            exact=exact,
            interval=interval,
            count=count,
            crop=crop,
        )
    ):
        name = (
            datetime.now().strftime("screenshot_%Y%m%d_%H%M%S_%f.png")
            if timestamped
            else f"screenshot_{index:04d}.png"
        )
        frame.save(destination / name)
