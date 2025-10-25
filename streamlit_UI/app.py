"""Streamlit front-end for the Echo policy calculator."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from policy_core import (
    BUFF_LABELS,
    BUFF_TYPE_COUNTS,
    BUFF_TYPES,
    DEFAULT_BUFF_WEIGHTS,
    MAX_SELECTED_TYPES,
    USER_COUNTS_JSON_PATH,
    build_active_counts,
    build_scorer,
    buff_names_to_indices,
    clone_count_maps,
    compute_optimal_policy,
    get_exp_refund_ratio,
    load_character_presets,
    load_user_buff_type_counts,
    make_cost_model,
    score_to_int,
    set_exp_refund_ratio,
    PolicyComputationResult,
    SimulationSummary,
)

ScoreFn = Callable[[int, float], float]

PLACEHOLDER = "选择词条"
NUM_SLOTS = MAX_SELECTED_TYPES
PRESET_CUSTOM_LABEL = "自定义"
PERCENT_BUFFS = {buff for buff in BUFF_TYPES if not buff.endswith("_Flat")}
USER_COUNTS_PATH = Path(__file__).resolve().parent / "user_counts_data.json"


def user_counts_available() -> bool:
    """Return True when the user counts JSON file exists."""

    return USER_COUNTS_PATH.exists()

CHARACTER_PRESET_PATH = Path(__file__).resolve().parent / "character_preset.json"
CHARACTER_BUFF_WEIGHT_PRESETS = load_character_presets(CHARACTER_PRESET_PATH)


def build_buff_value_options(buff_type_counts: Sequence[Mapping[int, int]]) -> dict[str, list[int]]:
    """Return selectable values for each buff type based on the provided counts."""

    return {
        buff: sorted(mapping.keys())
        for buff, mapping in zip(BUFF_TYPES, buff_type_counts)
    }


def format_buff_type_label(buff_name: str, weight_lookup: Mapping[str, float]) -> str:
    """Return the display label for a buff type, marking zero-weight entries."""

    if buff_name == PLACEHOLDER:
        return buff_name
    label = BUFF_LABELS.get(buff_name, buff_name)
    if weight_lookup.get(buff_name, 0.0) == 0.0:
        return f"（无效词条）{label}"
    return label


def format_value_label(buff_name: str, value: int) -> str:
    """Return a human-friendly label for a buff value."""

    if buff_name in PERCENT_BUFFS:
        return f"{value / 10:.1f}%"
    return str(value)


def reset_policy_results() -> None:
    """Clear cached policy results so UI reflects new inputs."""

    st.session_state.policy_result = None
    st.session_state.policy_error = None


def ensure_session_state_defaults() -> None:
    """Populate Streamlit session state with expected default entries."""

    if "buff_types" not in st.session_state:
        st.session_state.buff_types = [PLACEHOLDER] * NUM_SLOTS
    if "buff_values" not in st.session_state:
        st.session_state.buff_values = [None] * NUM_SLOTS
    if "buff_weights" not in st.session_state:
        st.session_state.buff_weights = DEFAULT_BUFF_WEIGHTS.copy()

    st.session_state.setdefault("policy_result", None)
    st.session_state.setdefault("policy_error", None)
    st.session_state.setdefault("selected_preset", PRESET_CUSTOM_LABEL)
    st.session_state.setdefault("last_applied_preset", PRESET_CUSTOM_LABEL)
    st.session_state.setdefault("exp_refund_ratio", float(get_exp_refund_ratio()))
    st.session_state.setdefault("target_score_input", 60.0)
    st.session_state.setdefault("cost_w_echo", 0.0)
    st.session_state.setdefault("cost_w_dkq", 1.0)
    st.session_state.setdefault("cost_w_exp", 0.0)
    st.session_state.setdefault("simulation_runs_input", 1_000_000)
    st.session_state.setdefault("simulation_seed_input", 42)
    st.session_state.setdefault("include_user_counts", False)
    st.session_state.setdefault("include_user_counts_prev", st.session_state.include_user_counts)


def update_weights_from_preset(selected_preset: str) -> bool:
    """Synchronise weight inputs with the selected preset, returning change status."""

    last_applied = st.session_state.last_applied_preset
    if selected_preset == last_applied:
        return False

    if selected_preset != PRESET_CUSTOM_LABEL:
        preset_weights = CHARACTER_BUFF_WEIGHT_PRESETS[selected_preset]
        st.session_state.buff_weights = {
            buff: float(preset_weights.get(buff, 0.0)) for buff in BUFF_TYPES
        }
        for buff, weight in st.session_state.buff_weights.items():
            st.session_state[f"buff_weight_widget_{buff}"] = weight

    st.session_state.last_applied_preset = selected_preset
    return True


def render_weight_inputs() -> bool:
    """Render weight controls and return True if any value changed."""

    weights_changed = False
    for buff_name in BUFF_TYPES:
        label_col, input_col = st.columns([1.4, 1.0], gap="small")
        label_col.markdown(
            f'<span class="weight-label">{BUFF_LABELS.get(buff_name, buff_name)}</span>',
            unsafe_allow_html=True,
        )
        widget_key = f"buff_weight_widget_{buff_name}"
        previous_value = float(st.session_state.buff_weights.get(buff_name, 0.0))
        if widget_key not in st.session_state:
            widget_kwargs = {"value": previous_value}
        else:
            widget_kwargs = {}
        weight_value = input_col.number_input(
            "Weight",
            key=widget_key,
            step=0.1,
            format="%g",
            label_visibility="collapsed",
            **widget_kwargs,
        )
        st.session_state.buff_weights[buff_name] = float(weight_value)
        if not math.isclose(weight_value, previous_value, rel_tol=1e-9, abs_tol=1e-9):
            weights_changed = True
    return weights_changed


def build_weight_lookup() -> dict[str, float]:
    """Return an ordered mapping from buff name to the current slider weight."""

    return {name: float(st.session_state.buff_weights.get(name, 0.0)) for name in BUFF_TYPES}


def render_buff_slots(
    scorer: Optional[ScoreFn],
    weight_lookup: Mapping[str, float],
    buff_value_options: Mapping[str, Sequence[int]],
) -> tuple[list[float], list[Optional[str]]]:
    """Render the buff slot selectors and return contributions and chosen names."""

    contributions: list[float] = []
    selected_buff_names: list[Optional[str]] = []

    if st.button("清空当前词条", type="secondary"):
        st.session_state.buff_types = [PLACEHOLDER] * NUM_SLOTS
        st.session_state.buff_values = [None] * NUM_SLOTS
        for idx in range(NUM_SLOTS):
            st.session_state[f"buff_type_widget_{idx}"] = PLACEHOLDER
            for suffix in ("", "_disabled", "_empty"):
                value_key = f"buff_value_widget_{idx}{suffix}"
                st.session_state.pop(value_key, None)

    for idx in range(NUM_SLOTS):
        index_col, type_col, value_col, score_col = st.columns([0.25, 2.0, 0.8, 0.7])

        with index_col:
            st.markdown(f"**#{idx + 1}**")

        current_type = st.session_state.buff_types[idx]
        taken_types = {
            st.session_state.buff_types[i]
            for i in range(NUM_SLOTS)
            if i != idx and st.session_state.buff_types[i] != PLACEHOLDER
        }
        allowed_types = [buff for buff in BUFF_TYPES if buff not in taken_types]
        allowed_types_sorted = sorted(
            allowed_types,
            key=lambda name: (-weight_lookup.get(name, 0.0), name),
        )
        type_choices = [PLACEHOLDER] + allowed_types_sorted

        if current_type not in type_choices:
            current_type = PLACEHOLDER
            st.session_state.buff_types[idx] = PLACEHOLDER

        type_index = type_choices.index(current_type) if current_type in type_choices else 0
        selected_type = type_col.selectbox(
            "Buff type",
            options=type_choices,
            index=type_index,
            key=f"buff_type_widget_{idx}",
            label_visibility="collapsed",
            format_func=lambda name: format_buff_type_label(name, weight_lookup),
        )

        st.session_state.buff_types[idx] = selected_type
        buff_name = selected_type if selected_type != PLACEHOLDER else None
        selected_buff_names.append(buff_name)

        current_value = st.session_state.buff_values[idx]

        if buff_name:
            value_options = list(buff_value_options.get(buff_name, []))
            if not value_options:
                st.session_state.buff_values[idx] = None
                value_col.selectbox(
                    "Value",
                    options=["-"],
                    index=0,
                    key=f"buff_value_widget_{idx}_empty",
                    disabled=True,
                    label_visibility="collapsed",
                )
                selected_value = None
            else:
                if current_value not in value_options:
                    current_value = value_options[0]
                    st.session_state.buff_values[idx] = current_value
                value_index = value_options.index(st.session_state.buff_values[idx])
                selected_value = value_col.selectbox(
                    "Value",
                    options=value_options,
                    index=value_index,
                    key=f"buff_value_widget_{idx}",
                    format_func=lambda v, name=buff_name: format_value_label(name, v),
                    label_visibility="collapsed",
                )
                st.session_state.buff_values[idx] = selected_value
        else:
            st.session_state.buff_values[idx] = None
            value_col.selectbox(
                "Value",
                options=["-"],
                index=0,
                key=f"buff_value_widget_{idx}_disabled",
                disabled=True,
                label_visibility="collapsed",
            )
            selected_value = None

        if buff_name and selected_value is not None and scorer:
            buff_index = BUFF_TYPES.index(buff_name)
            contribution = scorer(buff_index, float(selected_value))
            score_col.markdown(
                f"<div style='padding-top:0.4rem;'>Score {contribution:.2f}</div>",
                unsafe_allow_html=True,
            )
        else:
            contribution = 0.0
            score_col.markdown(
                "<div style='padding-top:0.4rem;'>Score —</div>",
                unsafe_allow_html=True,
            )
        contributions.append(contribution)

    enforce_selection_only_inputs()

    return contributions, selected_buff_names


def enforce_selection_only_inputs() -> None:
    """Prevent text entry on selectboxes to avoid triggering mobile keyboards."""

    components.html(
        """
        <script>
        const doc = window.parent.document;
        if (!window.parent.__echoSelectionOnlyPatch) {
            const disableTextEntry = () => {
                const inputs = doc.querySelectorAll('input[role="combobox"]');
                inputs.forEach((input) => {
                    input.setAttribute('inputmode', 'none');
                    input.setAttribute('readonly', 'true');
                    input.style.caretColor = 'transparent';
                });
            };
            disableTextEntry();
            const observer = new MutationObserver(disableTextEntry);
            observer.observe(doc.body, { childList: true, subtree: true });
            window.parent.__echoSelectionOnlyPatch = true;
        }
        </script>
        """,
        height=0,
    )


def render_total_score_card(
    total_score: float,
    selected_buff_names: Sequence[Optional[str]],
) -> None:
    """Display the aggregate score and current solver recommendation."""

    with st.container(border=True):
        st.metric("Total Score", f"{total_score:.2f}/100.00")
        st.markdown("**当前建议**")
        result = st.session_state.policy_result
        if isinstance(result, PolicyComputationResult):
            used_names = [name for name in selected_buff_names if name]
            used_indices = buff_names_to_indices(used_names)
            score_int = score_to_int(total_score)
            stage = len(used_indices)
            st.write(
                f"已揭示 {stage}/{NUM_SLOTS} 词条，目标分数 "
                f"{result.target_score:.2f}。"
            )
            if stage == 0:
                suggestion = "Continue"
            else:
                suggestion = result.solver.decision_output(used_indices, score_int)
            if suggestion == "Continue":
                st.success("建议继续")
            elif suggestion == "Abandon":
                st.error("建议放弃")
            else:
                st.info(suggestion)
        else:
            st.caption("计算策略后将显示强化建议。")


def render_policy_configuration(disable_compute: bool, counts_available: bool) -> bool:
    """Render policy configuration controls and return whether compute was requested."""

    with st.container(border=True):
        st.markdown('<div class="weights-card-title">策略参数</div>', unsafe_allow_html=True)

        st.markdown("**目标分数**")
        st.number_input(
            "强化目标分数",
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            key="target_score_input",
            label_visibility="collapsed",
            on_change=reset_policy_results,
        )

        include_user_counts = st.checkbox(
            "在计算中包含自定义统计数据",
            key="include_user_counts",
            help="勾选后会将自定义统计数据与内置数据合并，用于数值选择和策略计算。",
        )
        if not counts_available:
            st.warning("未找到自定义统计数据", icon="⚠️")

        st.markdown("**成本权重设定**")
        col_e, col_d, col_exp = st.columns(3)
        with col_e:
            st.caption("胚子成本权重")
            st.number_input(
                "胚子成本权重",
                step=0.1,
                format="%g",
                key="cost_w_echo",
                label_visibility="collapsed",
                on_change=reset_policy_results,
            )
        with col_d:
            st.caption("调谐器成本权重")
            st.number_input(
                "调谐器成本权重",
                step=0.1,
                format="%g",
                key="cost_w_dkq",
                label_visibility="collapsed",
                on_change=reset_policy_results,
            )
        with col_exp:
            st.caption("金密音筒成本权重")
            st.number_input(
                "金密音筒成本权重",
                step=0.1,
                format="%g",
                key="cost_w_exp",
                label_visibility="collapsed",
                on_change=reset_policy_results,
            )

        st.caption("强化失败经验返还比例(理想值:0.75)")
        exp_refund_ratio_value = st.slider(
            "强化失败经验返还比例",
            min_value=0.50,
            max_value=0.75,
            step=0.01,
            key="exp_refund_ratio",
            format="%.2f",
            label_visibility="collapsed",
            on_change=reset_policy_results,
        )
        set_exp_refund_ratio(float(exp_refund_ratio_value))

        st.markdown("**蒙特卡洛模拟设置(不影响策略)**")
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            st.caption("蒙特卡洛模拟次数")
            st.number_input(
                "Monte Carlo 样本数 (0=跳过)",
                min_value=0,
                max_value=5_000_000,
                step=1000,
                key="simulation_runs_input",
                label_visibility="collapsed",
            )
        with sim_col2:
            st.caption("随机数种子")
            st.number_input(
                "随机数种子",
                min_value=0,
                step=1,
                key="simulation_seed_input",
                label_visibility="collapsed",
            )

        return st.button(
            "开始计算策略",
            disabled=disable_compute,
            type="primary",
        )


def compute_policy(
    weight_lookup: Mapping[str, float],
    buff_type_counts: Sequence[Mapping[int, int]],
) -> None:
    """Compute the optimal policy with the current configuration."""

    st.session_state.policy_error = None
    st.session_state.policy_result = None
    try:
        cost_model = make_cost_model(
            w_echo=float(st.session_state.cost_w_echo),
            w_dkq=float(st.session_state.cost_w_dkq),
            w_exp=float(st.session_state.cost_w_exp),
        )
        with st.spinner("正在计算最优策略，这可能需要几分钟…"):
            result = compute_optimal_policy(
                buff_weights=weight_lookup,
                target_score=float(st.session_state.target_score_input),
                cost_model=cost_model,
                simulation_runs=int(st.session_state.simulation_runs_input),
                simulation_seed=int(st.session_state.simulation_seed_input),
                buff_type_counts=clone_count_maps(buff_type_counts),
            )
        st.session_state.policy_result = result
    except Exception as exc:  # broad to surface any numerical issues to the user
        st.session_state.policy_error = str(exc)


def render_policy_summary(result: PolicyComputationResult) -> None:
    """Render policy metrics and simulation tables in the Streamlit UI.

    Parameters
    ----------
    result:
        Dataclass bundle returned by ``compute_optimal_policy``.
    """

    with st.container(border=True):
        st.markdown("**策略计算结果**")
        col1, col2 = st.columns(2)
        col1.metric("λ*", f"{result.lambda_star:.8f}")
        col2.metric("期望成本", f"{result.expected_cost_per_success:.2f}")
        st.caption(f"DP 计算耗时 {result.compute_seconds:.2f} 秒")

        if isinstance(result.simulation, SimulationSummary):
            sim = result.simulation
            st.markdown("**蒙特卡洛模拟结果**")
            avg_cost = (
                result.cost_model.w_echo * sim.echo_per_success
                + result.cost_model.w_dkq * sim.dkq_per_success
                + result.cost_model.w_exp * sim.exp_per_success
            )
            full_upgrade_rate = (
                len(sim.max_slot_scores) / sim.total_runs if sim.total_runs > 0 else 0.0
            )

            success_cols = st.columns(2)
            success_cols[0].metric("成功率", f"{sim.success_rate * 100:.2f}%")
            success_cols[1].metric("满强化率", f"{full_upgrade_rate * 100:.2f}%")

            cost_cols = st.columns(4)
            cost_cols[0].metric("平均胚子消耗", f"{sim.echo_per_success:.2f}")
            cost_cols[1].metric("平均调谐器消耗", f"{sim.dkq_per_success:.2f}")
            cost_cols[2].metric("平均金密音筒消耗", f"{sim.exp_per_success:.2f}")
            cost_cols[3].metric("平均成本", f"{avg_cost:.2f}")

            if sim.max_slot_scores:
                scaled_scores = pd.Series([score / 100 for score in sim.max_slot_scores])
                bins = np.linspace(0, 100, 21, dtype=float)
                categories = pd.cut(
                    scaled_scores,
                    bins=bins,
                    include_lowest=True,
                    right=True,
                )
                intervals = categories.cat.categories
                counts = categories.value_counts().reindex(intervals, fill_value=0)
                total = counts.sum()
                if total > 0:
                    chart_data = pd.DataFrame(
                        {
                            "bin_start": [interval.left for interval in counts.index],
                            "bin_end": [interval.right for interval in counts.index],
                            "probability": counts.values / total,
                        }
                    )
                    histogram = alt.Chart(chart_data).mark_bar(
                        color="#6366f1",
                        opacity=0.9,
                        cornerRadiusTopLeft=2,
                        cornerRadiusTopRight=2,
                    ).encode(
                        x=alt.X(
                            "bin_start:Q",
                            title="满强化最终分数",
                            scale=alt.Scale(domain=(0, 100)),
                            axis=alt.Axis(
                                labelFontSize=11,
                                titleFontSize=12,
                                values=list(range(0, 101, 5)),
                                format=".0f",
                            ),
                        ),
                        x2="bin_end:Q",
                        y=alt.Y(
                            "probability:Q",
                            title="概率",
                            axis=alt.Axis(
                                format=".0%",
                                labelFontSize=11,
                                titleFontSize=12,
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("bin_start:Q", title="分数下限", format=".0f"),
                            alt.Tooltip("bin_end:Q", title="分数上限", format=".0f"),
                            alt.Tooltip("probability:Q", title="概率", format=".2%"),
                        ],
                    ).properties(height=240)
                    histogram = histogram.configure_view(strokeOpacity=0)
                    histogram = histogram.configure_axis(gridColor="#e2e8f0")
                    st.altair_chart(histogram, use_container_width=True)
                else:
                    st.caption("暂无满强化样本，已跳过满强化分布图。")
            else:
                st.caption("暂无满强化样本，已跳过满强化分布图。")

        if result.first_upgrade_table:
            st.markdown("**首条检视 (继续条件表)**")
            total_continue_prob = 0.0
            buff_slot_prob = 1.0 / len(BUFF_TYPES)
            for group in result.first_upgrade_table:
                label = BUFF_LABELS.get(group.buff_name, group.buff_name)
                options = sorted(group.options, key=lambda opt: opt.raw_value, reverse=True)
                if not options:
                    continue

                cumulative_prob = 0.0
                row_html_segments = []
                for opt in options:
                    single_prob = opt.probability * buff_slot_prob
                    cumulative_prob += single_prob
                    row_html_segments.append(
                        "<tr>"
                        f"<td>{format_value_label(group.buff_name, opt.raw_value)}</td>"
                        f"<td>{single_prob * 100:.2f}%</td>"
                        f"<td>{cumulative_prob * 100:.2f}%</td>"
                        "</tr>"
                    )

                total_continue_prob += cumulative_prob
                table_html = (
                    "<div class='prob-group'>"
                    f"<div class='prob-header'><span>{label}</span><span>累计概率 {cumulative_prob * 100:.2f}%</span></div>"
                    "<table class='prob-table'>"
                    "<thead><tr><th>数值</th><th>概率(%)</th><th>累计概率(%)</th></tr></thead>"
                    f"<tbody>{''.join(row_html_segments)}</tbody>"
                    "</table>"
                    "</div>"
                )
                st.markdown(table_html, unsafe_allow_html=True)
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            st.markdown(f"**总继续概率：{total_continue_prob * 100:.2f}%**")


def apply_page_styling() -> None:
    """Inject CSS tweaks that style the Streamlit app."""

    st.set_page_config(page_title="Echo Policy Calculator", layout="centered")
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #e3e6eb;
            border-radius: 12px;
            padding: 1.25rem;
            background-color: #ffffff;
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.25rem;
        }
        div[data-testid="stVerticalBlock"] div[data-testid="column"] {
            padding-top: 0.15rem;
            padding-bottom: 0.15rem;
        }
        .weights-card-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }
        .weight-label {
            display: flex;
            align-items: center;
            font-weight: 500;
            font-size: 0.95rem;
        }
        div[data-testid="stNumberInput"] > label {
            display: none;
        }
        div[data-testid="stNumberInput"] input {
            font-size: 1.05rem;
            padding: 0.35rem 0.6rem;
        }
        div[data-testid="stNumberInput"] input::-webkit-outer-spin-button,
        div[data-testid="stNumberInput"] input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        div[data-testid="stNumberInput"] input[type=number] {
            -moz-appearance: textfield;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 600;
            color: #0f172a;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.95rem;
            color: #475569;
        }
        .prob-group {
            margin-top: 0.85rem;
            padding: 0.6rem 0.8rem;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            background-color: #f8fafc;
        }
        .prob-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.4rem;
            font-weight: 600;
            color: #0f172a;
        }
        .prob-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }
        .prob-table th,
        .prob-table td {
            border: 1px solid #e2e8f0;
            padding: 0.35rem 0.55rem;
            text-align: left;
            white-space: nowrap;
        }
        .prob-table tbody tr:nth-child(odd) {
            background-color: #ffffff;
        }
        .prob-table tbody tr:nth-child(even) {
            background-color: #f1f5f9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Entry point used by Streamlit."""

    apply_page_styling()
    ensure_session_state_defaults()

    if st.session_state.include_user_counts != st.session_state.include_user_counts_prev:
        st.session_state.include_user_counts_prev = st.session_state.include_user_counts
        reset_policy_results()

    st.title("声骸强化策略计算器")
    counts_available = user_counts_available()

    with st.container(border=True):
        st.markdown('<div class="weights-card-title">权重设置</div>', unsafe_allow_html=True)
        preset_options = [PRESET_CUSTOM_LABEL] + sorted(CHARACTER_BUFF_WEIGHT_PRESETS.keys())
        st.caption("角色预设")
        selected_preset = st.selectbox(
            "角色预设",
            options=preset_options,
            key="selected_preset",
            label_visibility="collapsed",
        )
        weights_changed = update_weights_from_preset(selected_preset)
        weights_changed |= render_weight_inputs()

    if weights_changed:
        reset_policy_results()

    st.divider()

    include_user_counts = st.session_state.include_user_counts
    user_counts_data = (
        load_user_buff_type_counts(USER_COUNTS_PATH) if include_user_counts else None
    )
    active_counts = build_active_counts(include_user_counts, user_counts_data)
    buff_value_options = build_buff_value_options(active_counts)

    weight_lookup = build_weight_lookup()
    top_weights_sum = sum(sorted(weight_lookup.values(), reverse=True)[:NUM_SLOTS])
    if top_weights_sum <= 0:
        scorer: Optional[ScoreFn] = None
        st.warning("请输入至少一个大于 0 的权重以计算评分。")
    else:
        scorer = build_scorer(weight_lookup)

    contributions, selected_buff_names = render_buff_slots(
        scorer,
        weight_lookup,
        buff_value_options,
    )
    total_score = sum(contributions)
    render_total_score_card(total_score, selected_buff_names)

    compute_button = render_policy_configuration(
        disable_compute=scorer is None,
        counts_available=counts_available,
    )
    if compute_button and scorer is not None:
        compute_policy(weight_lookup, active_counts)

    if st.session_state.policy_error:
        st.error(f"策略计算失败：{st.session_state.policy_error}")
    elif isinstance(st.session_state.policy_result, PolicyComputationResult):
        render_policy_summary(st.session_state.policy_result)


if __name__ == "__main__":
    main()
