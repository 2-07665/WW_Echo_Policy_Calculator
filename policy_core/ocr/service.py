from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .common import ImageInput, ensure_pil_image
from .detector import PageMatchConfig, is_upgrade_page
from .engine import measure_last_call_duration
from .window_capture import CropRegion, WindowCapture, WindowCaptureConfig, WindowNotFoundError
from .workflow import BuffWorkflow, BuffWorkflowResult

if sys.platform != "win32":  # pragma: no cover - guard import usage
    raise ImportError("EchoUpgradeOCRService is only available on Windows.")


DEFAULT_DETECTION_CROP: CropRegion = (0.00, 0.00, 0.15, 0.10)
DEFAULT_OCR_CROP: CropRegion = (0.110, 0.29, 0.37, 0.50)
MIN_DETECTION_INTERVAL = 0.5


@dataclass
class OCRDebugInfo:
    window_found: bool = False
    on_upgrade_page: bool = False
    last_error: str | None = None
    last_capture_started: float | None = None
    last_ocr_duration: float | None = None


@dataclass(frozen=True)
class OCRStatus:
    active: bool
    debug: OCRDebugInfo = field(default_factory=OCRDebugInfo)
    result: BuffWorkflowResult | None = None
    result_timestamp: float | None = None


class EchoUpgradeOCRService:
    """
    Background worker that periodically captures the game upgrade UI and parses buff slots.
    """

    def __init__(
        self,
        *,
        window_title: str,
        workflow: BuffWorkflow | None = None,
        detection_crop: CropRegion = DEFAULT_DETECTION_CROP,
        ocr_crop: CropRegion = DEFAULT_OCR_CROP,
        detection_interval: float = 15.0,
        detailed_capture_delay: float = 0.1,
        upgrade_logo: ImageInput,
        page_match: PageMatchConfig | None = None,
    ) -> None:
        self.capture = WindowCapture(WindowCaptureConfig(title=window_title))
        self.workflow = workflow or BuffWorkflow()
        self.detection_crop = detection_crop
        self.ocr_crop = ocr_crop
        self.detection_interval = max(MIN_DETECTION_INTERVAL, float(detection_interval))
        self.detailed_capture_delay = max(0.0, float(detailed_capture_delay))
        self.upgrade_logo = ensure_pil_image(upgrade_logo)
        self.page_match = page_match or PageMatchConfig()

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest_result: BuffWorkflowResult | None = None
        self._latest_result_ts: float | None = None
        self._debug = OCRDebugInfo()

    def start(self, *, detection_interval: Optional[float] = None) -> None:
        with self._lock:
            if detection_interval is not None:
                self.detection_interval = max(MIN_DETECTION_INTERVAL, float(detection_interval))
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, name="EchoUpgradeOCR", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._thread:
                return
            self._stop_event.set()
            thread = self._thread
        thread.join(timeout=2.0)
        with self._lock:
            self._thread = None

    def status(self) -> OCRStatus:
        with self._lock:
            active = self._thread is not None and self._thread.is_alive()
            debug_copy = OCRDebugInfo(
                window_found=self._debug.window_found,
                on_upgrade_page=self._debug.on_upgrade_page,
                last_error=self._debug.last_error,
                last_capture_started=self._debug.last_capture_started,
            )
            result = self._latest_result
            timestamp = self._latest_result_ts
        return OCRStatus(active=active, debug=debug_copy, result=result, result_timestamp=timestamp)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            start_time = time.time()
            debug = OCRDebugInfo(last_capture_started=start_time)
            result: BuffWorkflowResult | None = None
            try:
                if not self.capture.window_exists():
                    debug.window_found = False
                    debug.last_error = "未找到游戏窗口，请确认鸣潮客户端已运行。"
                    debug.last_ocr_duration = None
                    self._record(debug, None, None)
                    self._wait_remaining(start_time)
                    continue
                debug.window_found = True
                detection_frame = self.capture.grab(crop=self.detection_crop)
                if not is_upgrade_page(detection_frame, self.upgrade_logo, self.page_match):
                    debug.on_upgrade_page = False
                    debug.last_error = "已找到窗口，但未检测到升级界面。"
                    debug.last_ocr_duration = None
                    self._record(debug, None, None)
                    self._wait_remaining(start_time)
                    continue
                debug.on_upgrade_page = True
                if self.detailed_capture_delay > 0:
                    if self._stop_event.wait(self.detailed_capture_delay):
                        break
                detailed_frame = self.capture.grab(crop=self.ocr_crop, restore=False)
                result = self.workflow.process_image(detailed_frame)
                debug.last_ocr_duration = measure_last_call_duration()
                debug.last_error = None
                self._record(debug, result, time.time())
            except WindowNotFoundError:
                debug.window_found = False
                debug.last_error = "未找到游戏窗口，请确认鸣潮客户端已运行。"
                debug.last_ocr_duration = None
                self._record(debug, None, None)
            except Exception as exc:  # noqa: BLE001
                debug.last_error = str(exc)
                debug.last_ocr_duration = None
                self._record(debug, None, None)
            self._wait_remaining(start_time)

    def _record(
        self,
        debug: OCRDebugInfo,
        result: BuffWorkflowResult | None,
        timestamp: float | None,
    ) -> None:
        with self._lock:
            self._debug = debug
            if result is not None:
                self._latest_result = result
                self._latest_result_ts = timestamp

    def _wait_remaining(self, start_time: float) -> None:
        elapsed = time.time() - start_time
        remaining = max(self.detection_interval - elapsed, MIN_DETECTION_INTERVAL / 2)
        self._stop_event.wait(remaining)
