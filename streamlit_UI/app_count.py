"""Streamlit helper UI for gathering buff type frequency data."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import sys
from typing import Final

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from policy_core import (
    BUFF_LABELS,
    BUFF_TYPE_COUNTS,
    BUFF_TYPES,
    empty_user_buff_type_counts,
    load_user_buff_type_counts,
    normalize_user_buff_type_counts,
    save_user_buff_type_counts,
)


COUNTS_FILE: Final[Path] = Path(__file__).resolve().parent / "user_counts_data.json"
BUFF_ROW_COLUMN_WIDTHS: Final[tuple[float, float, float]] = (1.5, 7.0, 1.5)
LAYOUT_STYLES: Final[str] = """
<style>
    div.block-container {
        padding-top: 1.5rem;
    }
    .stMarkdown p {
        margin-bottom: 0rem;
    }
    button[data-baseweb="button"] {
        padding: 0 0;
        font-size: 0.8rem;
        display: flex;
        flex-direction: column;
        gap: 0;
    }
    .buff-row-divider {
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
        margin: 0 0 0 0;
    }
</style>
"""


def default_user_counts() -> list[dict[int, int]]:
    """Return an empty counts structure aligned with BUFF_TYPES."""

    return empty_user_buff_type_counts()


def load_user_counts() -> list[dict[int, int]]:
    """Load user-maintained counts from disk, falling back to defaults."""

    if not COUNTS_FILE.exists():
        return default_user_counts()
    return load_user_buff_type_counts(COUNTS_FILE)


def save_user_counts(counts: list[dict[int, int]]) -> None:
    """Persist user-maintained counts to disk as JSON via the core helper."""

    normalized = normalize_user_buff_type_counts(counts)
    save_user_buff_type_counts(normalized, COUNTS_FILE)


def format_value_label(buff_name: str, value: int) -> str:
    """Return a display label for a stat value."""

    if buff_name.endswith("_Flat"):
        return str(value)
    return f"{value / 10:.1f}%"


def get_relative_counts_path() -> str:
    """Return a workspace-relative path for display."""

    try:
        return str(COUNTS_FILE.relative_to(Path.cwd()))
    except ValueError:
        return str(COUNTS_FILE)


def rerun_app() -> None:
    """Trigger a Streamlit rerun that works across Streamlit versions."""

    rerun_callable = getattr(st, "rerun", None)
    if callable(rerun_callable):
        rerun_callable()
        return

    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


def sync_user_counts() -> list[dict[int, int]]:
    """Synchronise session counts with disk and return the active mapping."""

    disk_counts = load_user_counts()
    session_counts = st.session_state.get("user_counts")

    if session_counts is None:
        st.session_state.user_counts = disk_counts
    else:
        st.session_state.user_counts = normalize_user_buff_type_counts(session_counts)
        if st.session_state.user_counts != disk_counts:
            st.session_state.user_counts = disk_counts

    return st.session_state.user_counts


def ensure_action_stack() -> None:
    """Ensure the undo stack exists in session state."""

    st.session_state.setdefault("action_stack", [])


def render_buff_row(
    buff_idx: int,
    buff_name: str,
    user_counts: list[dict[int, int]],
) -> None:
    """Render a single row of buff buttons."""

    extra_counts = user_counts[buff_idx]
    base_values = BUFF_TYPE_COUNTS[buff_idx].keys()
    combined_values = sorted(set(base_values) | set(extra_counts))
    display_name = BUFF_LABELS.get(buff_name, buff_name)

    name_col_left, values_col, name_col_right = st.columns(BUFF_ROW_COLUMN_WIDTHS, gap="small")
    name_col_left.markdown(f"**{display_name}**")
    name_col_right.markdown(
        f'<div style="text-align: right;"><strong>{display_name}</strong></div>',
        unsafe_allow_html=True,
    )

    if not combined_values:
        values_col.write("—")
        st.markdown('<div class="buff-row-divider"></div>', unsafe_allow_html=True)
        return

    for value, column in zip(combined_values, values_col.columns(len(combined_values))):
        extra_amount = extra_counts.get(value, 0)
        value_label = format_value_label(buff_name, value)
        button_label = f"{value_label} · {extra_amount}"

        with column:
            button_pressed = st.button(
                button_label,
                key=f"{buff_name}_{value}",
                help=f"{display_name}: {value_label}",
            )
            if button_pressed:
                extra_counts[value] = extra_amount + 1
                st.session_state.action_stack.append((buff_idx, value))
                save_user_counts(user_counts)
                st.session_state.user_counts = user_counts
                rerun_app()

    st.markdown('<div class="buff-row-divider"></div>', unsafe_allow_html=True)


def main() -> None:
    """Render the Streamlit application."""

    st.set_page_config(page_title="Buff Count Recorder", layout="wide")
    st.markdown(LAYOUT_STYLES, unsafe_allow_html=True)
    st.title("Buff Type Count Recorder")
    st.caption(
        "Click a stat button to record another observation. Totals combine the built-in data "
        "and your additions, which are stored in "
        f"`{get_relative_counts_path()}`."
    )

    counts_available = COUNTS_FILE.exists()
    if not counts_available:
        st.info(
            "No existing JSON data was found. The file will be created automatically when you record data.",
            icon="ℹ️",
        )

    ensure_action_stack()
    user_counts = sync_user_counts()

    undo_col, _, _ = st.columns([1, 1, 6], gap="large")
    if undo_col.button("Undo", type="secondary", disabled=not st.session_state.action_stack):
        last_action = st.session_state.action_stack.pop()
        buff_idx, value = last_action
        current = user_counts[buff_idx].get(value, 0)
        if current > 0:
            if current == 1:
                user_counts[buff_idx].pop(value)
            else:
                user_counts[buff_idx][value] = current - 1
            save_user_counts(user_counts)
        st.session_state.user_counts = user_counts
        rerun_app()

    for buff_idx, buff_name in enumerate(BUFF_TYPES):
        render_buff_row(buff_idx, buff_name, user_counts)


if __name__ == "__main__":
    main()
