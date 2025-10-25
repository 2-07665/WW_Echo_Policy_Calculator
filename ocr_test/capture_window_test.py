from __future__ import annotations

import win32gui

from capture_window import capture_frames, save_frames
from ocr import extract_text

# Update these constants to match your target window and capture preferences.
WINDOW_TITLE = "鸣潮  "
CROP_REGION = (0.00, 0.00, 0.15, 0.1)
CAPTURE_INTERVAL = 1.0
CAPTURE_COUNT = 1

def main() -> None:

    save_frames(
            title=WINDOW_TITLE,
            interval=CAPTURE_INTERVAL,
            count=CAPTURE_COUNT,
            crop=CROP_REGION,
        )


if __name__ == "__main__":
    main()
