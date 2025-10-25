"""Shared service layer for UI front-ends."""

from __future__ import annotations

import os
import sys
import threading
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from policy_core import (  # noqa: F401
    BUFF_LABELS,
    BUFF_TYPE_COUNTS,
    BUFF_TYPE_MAX_VALUES,
    BUFF_TYPES,
    DEFAULT_BUFF_WEIGHTS,
    MAX_SELECTED_TYPES,
    PolicyComputationResult,
    SimulationSummary,
    buff_names_to_indices,
    compute_optimal_policy,
    get_exp_refund_ratio,
    make_cost_model,
    score_to_int,
    set_exp_refund_ratio,
)
from policy_core import (
    USER_COUNTS_JSON_PATH,
    build_active_counts,
    clone_count_maps,
    load_character_presets,
    load_user_buff_type_counts,
)

IS_WINDOWS = sys.platform == "win32"
HAVE_OCR_DEPS = False

if IS_WINDOWS:  # pragma: no cover - Windows-only dependency surface
    try:
        from PIL import Image

        from policy_core.ocr.service import (
            DEFAULT_DETECTION_CROP,
            DEFAULT_OCR_CROP,
            EchoUpgradeOCRService,
            OCRStatus,
        )
    except ImportError:  # pragma: no cover - optional dependency guard
        EchoUpgradeOCRService = None  # type: ignore[assignment]
        Image = None  # type: ignore[assignment]
        OCRStatus = None  # type: ignore[assignment]
        HAVE_OCR_DEPS = False
    else:
        HAVE_OCR_DEPS = True
else:  # pragma: no cover - non-Windows build guard
    EchoUpgradeOCRService = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    OCRStatus = None  # type: ignore[assignment]


def serialize_simulation(simulation: SimulationSummary | None) -> Any:
    """Convert a simulation summary into a JSON-friendly payload."""

    if simulation is None:
        return None
    return {
        "success_rate": simulation.success_rate,
        "echo_per_success": simulation.echo_per_success,
        "dkq_per_success": simulation.dkq_per_success,
        "exp_per_success": simulation.exp_per_success,
        "max_slot_scores": simulation.max_slot_scores,
        "total_runs": simulation.total_runs,
    }


def serialize_first_upgrade_table(result: PolicyComputationResult) -> list[dict[str, Any]]:
    """Return the first-upgrade probability table in a serialisable form."""

    table: list[dict[str, Any]] = []
    for group in result.first_upgrade_table:
        table.append(
            {
                "buff_name": group.buff_name,
                "options": [
                    {
                        "raw_value": option.raw_value,
                        "score": option.score,
                        "probability": option.probability,
                    }
                    for option in group.options
                ],
            }
        )
    return table


def serialize_result(result: PolicyComputationResult) -> dict[str, Any]:
    """Serialise a policy computation result for UI consumption."""

    return {
        "target_score": result.target_score,
        "lambda_star": result.lambda_star,
        "expected_cost_per_success": result.expected_cost_per_success,
        "compute_seconds": result.compute_seconds,
        "cost_model": {
            "w_echo": result.cost_model.w_echo,
            "w_dkq": result.cost_model.w_dkq,
            "w_exp": result.cost_model.w_exp,
        },
        "simulation": serialize_simulation(result.simulation),
        "first_upgrade_table": serialize_first_upgrade_table(result),
    }


def serialize_ocr_status(status: OCRStatus | None) -> dict[str, Any]:
    """Convert OCR status objects into a JSON-friendly shape."""

    if not HAVE_OCR_DEPS:
        return {
            "active": False,
            "debug": {"last_ocr_duration": None},
            "result": None,
            "result_timestamp": None,
        }
    if status is None:
        return {
            "active": False,
            "debug": {
                "window_found": False,
                "on_upgrade_page": False,
                "last_error": None,
                "last_capture_started": None,
                "last_ocr_duration": None,
            },
            "result": None,
            "result_timestamp": None,
        }
    result_payload: dict[str, Any] | None = None
    if status.result is not None:
        result_payload = status.result.to_ui_payload()
    return {
        "active": status.active,
        "debug": {
            "window_found": status.debug.window_found,
            "on_upgrade_page": status.debug.on_upgrade_page,
            "last_error": status.debug.last_error,
            "last_capture_started": status.debug.last_capture_started,
            "last_ocr_duration": status.debug.last_ocr_duration,
        },
        "result": result_payload,
        "result_timestamp": status.result_timestamp,
    }


class PolicyUIService:
    """Application service that exposes calculator operations to UI layers."""

    def __init__(
        self,
        *,
        preset_path: str | os.PathLike[str] | None = None,
        user_counts_path: str | os.PathLike[str] | None = None,
        enable_ocr: bool = True,
    ) -> None:
        self._results: dict[str, PolicyComputationResult] = {}
        self._results_lock = threading.Lock()
        self._next_result_id = 1

        self._character_presets = load_character_presets(preset_path)
        self._user_counts_path = Path(user_counts_path) if user_counts_path else USER_COUNTS_JSON_PATH

        self._ocr_enabled = bool(enable_ocr and HAVE_OCR_DEPS)
        self._ocr_lock = threading.Lock()
        self._ocr_service: EchoUpgradeOCRService | None = None
        self._ocr_logo: Image.Image | None = None  # type: ignore[assignment]

    # ---- Core policy operations -------------------------------------------------

    def bootstrap(self) -> dict[str, Any]:
        """Return static data required to initialise the UI."""

        return {
            "buff_types": BUFF_TYPES,
            "buff_labels": BUFF_LABELS,
            "buff_type_counts": clone_count_maps(BUFF_TYPE_COUNTS),
            "user_buff_type_counts": clone_count_maps(
                load_user_buff_type_counts(self._user_counts_path)
            ),
            "user_counts_available": self._user_counts_path.exists(),
            "user_counts_path": str(self._user_counts_path),
            "buff_type_max_values": list(BUFF_TYPE_MAX_VALUES),
            "default_buff_weights": dict(DEFAULT_BUFF_WEIGHTS),
            "max_selected_types": MAX_SELECTED_TYPES,
            "presets": self._character_presets,
            "exp_refund_ratio": get_exp_refund_ratio(),
        }

    def set_exp_refund_ratio(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Persist a new experience refund ratio and echo back the stored value."""

        value = float(payload.get("value", 0.0))
        set_exp_refund_ratio(value)
        return {"value": get_exp_refund_ratio()}

    def compute_policy(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Compute the optimal policy and retain the result for follow-up queries."""

        buff_weights = payload.get("buff_weights", {})
        target_score = float(payload.get("target_score", 0.0))
        simulation_runs = int(payload.get("simulation_runs", 0))
        simulation_seed = int(payload.get("simulation_seed", 42))
        include_user_counts = bool(payload.get("include_user_counts", False))
        cost_weights = payload.get("cost_weights", {})

        w_echo = float(cost_weights.get("w_echo", 0.0))
        w_dkq = float(cost_weights.get("w_dkq", 0.0))
        w_exp = float(cost_weights.get("w_exp", 0.0))

        user_counts = load_user_buff_type_counts(self._user_counts_path)
        active_counts = build_active_counts(include_user_counts, user_counts)
        cost_model = make_cost_model(w_echo=w_echo, w_dkq=w_dkq, w_exp=w_exp)
        result = compute_optimal_policy(
            buff_weights=buff_weights,
            target_score=target_score,
            cost_model=cost_model,
            simulation_runs=simulation_runs,
            simulation_seed=simulation_seed,
            buff_type_counts=active_counts,
        )

        with self._results_lock:
            result_id = str(self._next_result_id)
            self._next_result_id += 1
            self._results[result_id] = result

        return {
            "result_id": result_id,
            "summary": serialize_result(result),
        }

    def policy_suggestion(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return a keep/continue suggestion for the supplied buff set."""

        result_id = payload.get("result_id")
        if not result_id:
            raise ValueError("Missing result_id; please compute a policy first.")

        with self._results_lock:
            result = self._results.get(str(result_id))

        if result is None:
            raise ValueError("Unknown result_id; please compute a policy first.")

        names = [name for name in payload.get("buff_names", []) if name]
        score_value = float(payload.get("total_score", 0.0))

        try:
            indices = buff_names_to_indices(names)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        if len(indices) == 0:
            suggestion = "Continue"
        else:
            suggestion = result.solver.decision_output(indices, score_to_int(score_value))

        return {
            "suggestion": suggestion,
            "stage": len(indices),
            "target_score": result.target_score,
        }

    # ---- OCR operations ---------------------------------------------------------

    def ocr_capabilities(self, _payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Expose whether OCR is available on the current platform."""

        return {"supported": self._ocr_enabled}

    def start_ocr(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Ensure the OCR service is running (Windows only)."""

        if not self._ocr_enabled:
            status = serialize_ocr_status(None)
            status["detection_interval"] = None
            return {"supported": False, "status": status}

        service = self._ensure_ocr_service()
        interval = None
        if payload:
            raw_interval = payload.get("detection_interval")
            if raw_interval is not None:
                try:
                    interval = float(raw_interval)
                except (TypeError, ValueError) as exc:
                    raise ValueError("detection_interval must be numeric.") from exc

        service.start(detection_interval=interval)
        status = serialize_ocr_status(service.status())
        status["detection_interval"] = service.detection_interval
        return {"supported": True, "status": status}

    def stop_ocr(self, _payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Stop the OCR loop when running on Windows."""

        if not self._ocr_enabled:
            status = serialize_ocr_status(None)
            status["detection_interval"] = None
            return {"supported": False, "status": status}

        service = self._ensure_ocr_service()
        service.stop()
        status = serialize_ocr_status(service.status())
        status["detection_interval"] = service.detection_interval
        return {"supported": True, "status": status}

    def poll_ocr_status(self, _payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return the most recent OCR status snapshot."""

        if not self._ocr_enabled:
            status = serialize_ocr_status(None)
            status["detection_interval"] = None
            return {"supported": False, "status": status}

        with self._ocr_lock:
            service = self._ocr_service

        if service is None:
            default_status = serialize_ocr_status(None)
            default_status["detection_interval"] = float(
                os.environ.get("OCR_DETECTION_INTERVAL", "15.0")
            )
            return {"supported": True, "status": default_status}

        status = serialize_ocr_status(service.status())
        status["detection_interval"] = service.detection_interval
        return {"supported": True, "status": status}

    # ---- Platform info ---------------------------------------------------------

    @staticmethod
    def get_platform() -> str:
        """Return the host platform identifier (matches sys.platform)."""

        return sys.platform

    # ---- Helpers ----------------------------------------------------------------

    def _ensure_ocr_service(self) -> EchoUpgradeOCRService:
        if not self._ocr_enabled:
            raise RuntimeError("OCR service is disabled on this platform.")

        with self._ocr_lock:
            if self._ocr_service is None:
                window_title = os.environ.get("OCR_WINDOW_TITLE", "鸣潮  ")
                detection_interval = float(os.environ.get("OCR_DETECTION_INTERVAL", "15.0"))
                detailed_interval = float(os.environ.get("OCR_DETAILED_INTERVAL", "0.1"))
                logo = self._load_upgrade_logo()
                self._ocr_service = EchoUpgradeOCRService(
                    window_title=window_title,
                    detection_crop=DEFAULT_DETECTION_CROP,
                    ocr_crop=DEFAULT_OCR_CROP,
                    detection_interval=detection_interval,
                    detailed_capture_delay=detailed_interval,
                    upgrade_logo=logo,
                )
        # mypy assertion helper: _ocr_service must be initialised at this point
        assert self._ocr_service is not None
        return self._ocr_service

    def _load_upgrade_logo(self) -> Image.Image:
        if not self._ocr_enabled:
            raise RuntimeError("OCR assets are unavailable when OCR is disabled.")

        with self._ocr_lock:
            if self._ocr_logo is not None:
                return self._ocr_logo

            logo_resource = resources.files("policy_core.ocr").joinpath("upgrade_page_logo.png")
            if not logo_resource.is_file():
                raise FileNotFoundError("Upgrade page logo not found in policy_core.ocr resources")

            with logo_resource.open("rb") as stream:  # type: ignore[attr-defined]
                with Image.open(stream) as loaded:  # type: ignore[arg-type]
                    self._ocr_logo = loaded.convert("RGB")
                    return self._ocr_logo
