(() => {
  const PLACEHOLDER_LABEL = '选择词条';
  const PRESET_CUSTOM_LABEL = '自定义';

  const state = {
    buffTypes: [],
    buffLabels: {},
    buffTypeCounts: [],
    userBuffTypeCounts: [],
    buffTypeMaxValues: [],
    maxSelectedTypes: 5,
    defaultBuffWeights: {},
    presets: {},
    weightMap: {},
    selectedPreset: PRESET_CUSTOM_LABEL,
    activeCounts: [],
    buffValueOptions: new Map(),
    buffSelections: [],
    buffValues: [],
    contributions: [],
    percentBuffs: new Set(),
    buffIndex: new Map(),
    topWeightsSum: 0,
    scorer: null,
    totalScore: 0,
    policySummary: null,
    policyError: null,
    resultId: null,
    suggestion: null,
    suggestionPending: false,
    expRefundRatio: 0.66,
    targetScore: 60.0,
    costWeights: { wEcho: 0.0, wDkq: 1.0, wExp: 0.0 },
    simulationRuns: 1000000,
    simulationSeed: 42,
    includeUserCounts: false,
    userCountsAvailable: false,
    userCountsPath: '',
  };

  const elements = {};
  const weightInputRefs = new Map();
  let suggestionRequestToken = 0;

  function getPywebviewApi() {
    if (window.pywebview && window.pywebview.api) {
      return window.pywebview.api;
    }
    return null;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function convertCountsList(rawList) {
    if (!Array.isArray(rawList)) {
      return [];
    }
    return rawList.map((mapping) => {
      const converted = {};
      for (const [key, amount] of Object.entries(mapping)) {
        converted[Number(key)] = Number(amount);
      }
      return converted;
    });
  }

  function cloneCountsList(counts) {
    return counts.map((mapping) => ({ ...mapping }));
  }

  async function init() {
    cacheElements();
    try {
      const data = await window.api.bootstrap();
      initialiseState(data);
      setupPresetOptions();
      renderWeightInputs();
      setupEventHandlers();
      renderUserCountsHint();
      updateActiveCounts();
      computeScorer();
      computeContributions();
      renderBuffSlots();
      renderTotalScoreCard();
      updateComputeButtonState();
      renderResults();
    } catch (error) {
      console.error('Failed to bootstrap Electron UI', error);
      elements.resultsSection.innerHTML = `
        <div class="error-message">初始化失败：${escapeHtml(error.message || error)}</div>
      `;
    }
  }

  function cacheElements() {
    elements.presetSelect = document.getElementById('preset-select');
    elements.weightInputsContainer = document.getElementById('weight-inputs');
    elements.buffSlotsContainer = document.getElementById('buff-slots');
    elements.clearBuffsButton = document.getElementById('clear-buffs');
    elements.scoreCard = document.getElementById('score-card');
    elements.targetScoreInput = document.getElementById('target-score-input');
    elements.includeUserCountsToggle = document.getElementById('include-user-counts');
    elements.userCountsHint = document.getElementById('user-counts-hint');
    elements.costWEchoInput = document.getElementById('cost-w-echo');
    elements.costWDkqInput = document.getElementById('cost-w-dkq');
    elements.costWExpInput = document.getElementById('cost-w-exp');
    elements.expRefundSlider = document.getElementById('exp-refund-slider');
    elements.expRefundValue = document.getElementById('exp-refund-value');
    elements.simulationRunsInput = document.getElementById('simulation-runs-input');
    elements.simulationSeedInput = document.getElementById('simulation-seed-input');
    elements.computeButton = document.getElementById('compute-button');
    elements.resultsSection = document.getElementById('results-section');
  }

  async function resolvePlatform() {
    if (!window.api) {
      return navigator.platform.startsWith('Win') ? 'win32' : 'unknown';
    }
    if (typeof window.api.platform === 'string' && window.api.platform) {
      return window.api.platform;
    }
    if (window.api.platform && typeof window.api.platform.then === 'function') {
      try {
        return await window.api.platform;
      } catch {
        /* ignore */
      }
    }
    if (typeof window.api.getPlatform === 'function') {
      try {
        const value = await window.api.getPlatform();
        window.api.platform = value;
        return value;
      } catch {
        /* ignore */
      }
    }
    if (
      window.pywebview &&
      window.pywebview.api &&
      typeof window.pywebview.api.get_platform === 'function'
    ) {
      try {
        const value = await window.pywebview.api.get_platform();
        window.api.platform = value;
        return value;
      } catch {
        /* ignore */
      }
    }
    return navigator.platform.startsWith('Win') ? 'win32' : 'unknown';
  }

  function initialiseState(data) {
    state.buffTypes = data.buff_types;
    state.buffLabels = data.buff_labels;
    state.buffTypeCounts = convertCountsList(data.buff_type_counts);
    state.userBuffTypeCounts = convertCountsList(data.user_buff_type_counts);
    state.buffTypeMaxValues = data.buff_type_max_values.map(Number);
    state.maxSelectedTypes = Number(data.max_selected_types);
    state.defaultBuffWeights = data.default_buff_weights;
    state.presets = data.presets;
    state.expRefundRatio = Number(data.exp_refund_ratio);
    state.userCountsAvailable = Boolean(data.user_counts_available);
    state.userCountsPath = data.user_counts_path || '';
    state.includeUserCounts = false;

    state.buffIndex = new Map(state.buffTypes.map((name, idx) => [name, idx]));
    state.percentBuffs = new Set(state.buffTypes.filter((name) => !name.endsWith('_Flat')));

    state.weightMap = {};
    state.buffTypes.forEach((name) => {
      state.weightMap[name] = Number(state.defaultBuffWeights[name] ?? 0);
    });

    state.selectedPreset = PRESET_CUSTOM_LABEL;
    state.buffSelections = Array(state.maxSelectedTypes).fill(null);
    state.buffValues = Array(state.maxSelectedTypes).fill(null);
    state.contributions = Array(state.maxSelectedTypes).fill(0);
    state.totalScore = 0;
    state.policySummary = null;
    state.policyError = null;
    state.resultId = null;
    state.suggestion = null;
    state.suggestionPending = false;

    elements.targetScoreInput.value = state.targetScore;
    elements.costWEchoInput.value = state.costWeights.wEcho;
    elements.costWDkqInput.value = state.costWeights.wDkq;
    elements.costWExpInput.value = state.costWeights.wExp;
    elements.expRefundSlider.value = state.expRefundRatio.toFixed(2);
    elements.expRefundValue.textContent = state.expRefundRatio.toFixed(2);
    elements.simulationRunsInput.value = state.simulationRuns;
    elements.simulationSeedInput.value = state.simulationSeed;
    elements.includeUserCountsToggle.checked = state.includeUserCounts;
  }

  function setupPresetOptions() {
    const select = elements.presetSelect;
    const fragment = document.createDocumentFragment();
    const customOption = document.createElement('option');
    customOption.value = PRESET_CUSTOM_LABEL;
    customOption.textContent = PRESET_CUSTOM_LABEL;
    fragment.appendChild(customOption);

    Object.keys(state.presets)
      .sort((a, b) => a.localeCompare(b, 'zh-Hans-CN'))
      .forEach((name) => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        fragment.appendChild(option);
      });
    select.innerHTML = '';
    select.appendChild(fragment);
    select.value = PRESET_CUSTOM_LABEL;
  }

  function renderWeightInputs() {
    const layout = [
      ['Crit_Rate', 'Crit_Damage'],
      ['Attack', 'Attack_Flat'],
      ['HP', 'HP_Flat'],
      ['Defence', 'Defence_Flat'],
      ['Basic_Attack_Damage', 'Heavy_Attack_Damage'],
      ['Skill_Damage', 'Ult_Damage'],
      ['ER'],
    ];

    const container = elements.weightInputsContainer;
    container.innerHTML = '';
    weightInputRefs.clear();

    const available = new Set(state.buffTypes);
    const seen = new Set();

    const appendField = (buffName) => {
      if (!available.has(buffName)) {
        return;
      }
      seen.add(buffName);
      const wrapper = document.createElement('div');
      wrapper.className = 'weight-field';

      const label = document.createElement('span');
      label.textContent = state.buffLabels[buffName] ?? buffName;

      const input = document.createElement('input');
      input.type = 'number';
      input.step = '0.1';
      input.value = Number(state.weightMap[buffName] ?? 0);

      input.addEventListener('change', () => {
        const value = Number.isFinite(input.valueAsNumber) ? input.valueAsNumber : 0;
        if (state.weightMap[buffName] === value) {
          return;
        }
        state.weightMap[buffName] = value;
        if (state.selectedPreset !== PRESET_CUSTOM_LABEL) {
          state.selectedPreset = PRESET_CUSTOM_LABEL;
          elements.presetSelect.value = PRESET_CUSTOM_LABEL;
        }
        onWeightsUpdated();
      });

      wrapper.appendChild(label);
      wrapper.appendChild(input);
      container.appendChild(wrapper);
      weightInputRefs.set(buffName, input);
    };

    layout.forEach((row) => {
      row.forEach(appendField);
      if (row.length === 1) {
        const placeholder = document.createElement('div');
        placeholder.className = 'weight-field weight-placeholder';
        container.appendChild(placeholder);
      }
    });

    state.buffTypes.forEach((buffName) => {
      if (!seen.has(buffName)) {
        appendField(buffName);
      }
    });
  }

  function setupEventHandlers() {
    elements.presetSelect.addEventListener('change', () => {
      applyPreset(elements.presetSelect.value);
    });

    elements.clearBuffsButton.addEventListener('click', () => {
      state.buffSelections.fill(null);
      state.buffValues.fill(null);
      state.contributions.fill(0);
      state.totalScore = 0;
      renderBuffSlots();
      renderTotalScoreCard();
      updateComputeButtonState();
      if (state.resultId) {
        updateSuggestion();
      }
    });

    elements.targetScoreInput.addEventListener('change', () => {
      const value = clampNumber(elements.targetScoreInput.valueAsNumber, 0.1, 100.0, state.targetScore);
      state.targetScore = value;
      elements.targetScoreInput.value = value;
      resetPolicyResult();
    });

    elements.includeUserCountsToggle.addEventListener('change', () => {
      state.includeUserCounts = Boolean(elements.includeUserCountsToggle.checked);
      updateActiveCounts();
      computeContributions();
      renderBuffSlots();
      resetPolicyResult();
      updateComputeButtonState();
      renderUserCountsHint();
    });

    elements.costWEchoInput.addEventListener('change', () => {
      state.costWeights.wEcho = Number.isFinite(elements.costWEchoInput.valueAsNumber)
        ? elements.costWEchoInput.valueAsNumber
        : state.costWeights.wEcho;
      elements.costWEchoInput.value = state.costWeights.wEcho;
      resetPolicyResult();
    });

    elements.costWDkqInput.addEventListener('change', () => {
      state.costWeights.wDkq = Number.isFinite(elements.costWDkqInput.valueAsNumber)
        ? elements.costWDkqInput.valueAsNumber
        : state.costWeights.wDkq;
      elements.costWDkqInput.value = state.costWeights.wDkq;
      resetPolicyResult();
    });

    elements.costWExpInput.addEventListener('change', () => {
      state.costWeights.wExp = Number.isFinite(elements.costWExpInput.valueAsNumber)
        ? elements.costWExpInput.valueAsNumber
        : state.costWeights.wExp;
      elements.costWExpInput.value = state.costWeights.wExp;
      resetPolicyResult();
    });

    elements.expRefundSlider.addEventListener('input', () => {
      elements.expRefundValue.textContent = Number(elements.expRefundSlider.value).toFixed(2);
    });

    elements.expRefundSlider.addEventListener('change', async () => {
      const ratio = Number(elements.expRefundSlider.value);
      try {
        const response = await window.api.setExpRefundRatio(ratio);
        state.expRefundRatio = Number(response.value);
        elements.expRefundSlider.value = state.expRefundRatio.toFixed(2);
        elements.expRefundValue.textContent = state.expRefundRatio.toFixed(2);
        resetPolicyResult();
      } catch (error) {
        console.error('Failed to update EXP refund ratio', error);
      }
    });

    elements.simulationRunsInput.addEventListener('change', () => {
      const value = clampNumber(elements.simulationRunsInput.valueAsNumber, 0, 5_000_000, state.simulationRuns);
      state.simulationRuns = Math.round(value);
      elements.simulationRunsInput.value = state.simulationRuns;
    });

    elements.simulationSeedInput.addEventListener('change', () => {
      const value = clampNumber(elements.simulationSeedInput.valueAsNumber, 0, Number.MAX_SAFE_INTEGER, state.simulationSeed);
      state.simulationSeed = Math.round(value);
      elements.simulationSeedInput.value = state.simulationSeed;
    });

    elements.computeButton.addEventListener('click', () => {
      if (!state.scorer || elements.computeButton.disabled) {
        return;
      }
      handleCompute();
    });

  }

  function clampNumber(value, min, max, fallback) {
    if (!Number.isFinite(value)) {
      return fallback;
    }
    return Math.min(Math.max(value, min), max);
  }

  function applyPreset(presetName) {
    if (presetName === PRESET_CUSTOM_LABEL) {
      state.selectedPreset = PRESET_CUSTOM_LABEL;
      return;
    }

    const preset = state.presets[presetName];
    if (!preset) {
      return;
    }
    state.selectedPreset = presetName;

    state.buffTypes.forEach((buffName) => {
      const value = Number(preset[buffName] ?? 0);
      state.weightMap[buffName] = value;
      const input = weightInputRefs.get(buffName);
      if (input) {
        input.value = value;
      }
    });

    onWeightsUpdated();
  }

  function onWeightsUpdated() {
    computeScorer();
    computeContributions();
    renderBuffSlots();
    renderTotalScoreCard();
    updateComputeButtonState();
    resetPolicyResult();
  }

  function updateActiveCounts() {
    const baseCounts = cloneCountsList(state.buffTypeCounts);
    if (state.includeUserCounts && state.userBuffTypeCounts.length) {
      for (let idx = 0; idx < state.buffTypes.length; idx += 1) {
        const merged = { ...(baseCounts[idx] ?? {}) };
        const extra = state.userBuffTypeCounts[idx] ?? {};
        for (const [rawValue, rawAmount] of Object.entries(extra)) {
          const value = Number(rawValue);
          const amount = Number(rawAmount);
          merged[value] = (merged[value] ?? 0) + amount;
        }
        baseCounts[idx] = merged;
      }
    }

    state.activeCounts = baseCounts;
    state.buffValueOptions = new Map(
      state.buffTypes.map((name, idx) => {
        const mapping = state.activeCounts[idx] ?? {};
        const values = Object.keys(mapping)
          .map(Number)
          .sort((a, b) => a - b);
        return [name, values];
      }),
    );
    normalizeSelections();
  }

  function renderUserCountsHint() {
    if (!elements.userCountsHint) {
      return;
    }
    elements.userCountsHint.classList.remove('warning');
    if (!state.userCountsAvailable) {
      elements.userCountsHint.textContent = '未找到自定义统计数据';
      elements.userCountsHint.classList.add('warning');
      return;
    }
    elements.userCountsHint.innerHTML = '';
  }

  function normalizeSelections() {
    const seen = new Set();
    for (let idx = 0; idx < state.buffSelections.length; idx += 1) {
      const currentType = state.buffSelections[idx];
      if (!currentType || !state.buffTypes.includes(currentType) || seen.has(currentType)) {
        state.buffSelections[idx] = null;
        state.buffValues[idx] = null;
        continue;
      }
      seen.add(currentType);
      const valueOptions = state.buffValueOptions.get(currentType) ?? [];
      if (!valueOptions.length) {
        state.buffValues[idx] = null;
        continue;
      }
      if (!valueOptions.includes(state.buffValues[idx])) {
        state.buffValues[idx] = valueOptions[0];
      }
    }
  }

  function computeScorer() {
    const weightList = state.buffTypes.map((name) => Number(state.weightMap[name] ?? 0));
    const sortedWeights = [...weightList].sort((a, b) => b - a);
    const topWeights = sortedWeights.slice(0, state.maxSelectedTypes);
    const topSum = topWeights.reduce((acc, value) => acc + value, 0);
    state.topWeightsSum = topSum;

    if (topSum <= 0) {
      state.scorer = null;
      return;
    }

    state.scorer = (buffName, value) => {
      const idx = state.buffIndex.get(buffName);
      if (idx === undefined) {
        return 0;
      }
      const weight = Number(state.weightMap[buffName] ?? 0);
      const maxValue = Number(state.buffTypeMaxValues[idx] ?? 1);
      if (maxValue <= 0) {
        return 0;
      }
      return (100 * weight * Number(value)) / (topSum * maxValue);
    };
  }

  function computeContributions() {
    const contributions = [];
    let total = 0;
    for (let idx = 0; idx < state.maxSelectedTypes; idx += 1) {
      const buffName = state.buffSelections[idx];
      const rawValue = state.buffValues[idx];
      let contribution = 0;
      if (buffName && rawValue != null && state.scorer) {
        contribution = state.scorer(buffName, rawValue);
      }
      contributions[idx] = contribution;
      total += contribution;
    }
    state.contributions = contributions;
    state.totalScore = total;
  }

  function renderBuffSlots() {
    normalizeSelections();
    const container = elements.buffSlotsContainer;
    const rows = [];

    for (let idx = 0; idx < state.maxSelectedTypes; idx += 1) {
      const currentType = state.buffSelections[idx];
      const currentValue = state.buffValues[idx];
      const contribution = state.contributions[idx] ?? 0;

      const taken = new Set();
      state.buffSelections.forEach((name, innerIdx) => {
        if (innerIdx !== idx && name) {
          taken.add(name);
        }
      });

      const available = state.buffTypes
        .filter((name) => !taken.has(name))
        .sort((a, b) => {
          const weightDiff = Number(state.weightMap[b] ?? 0) - Number(state.weightMap[a] ?? 0);
          if (Math.abs(weightDiff) > 1e-9) {
            return weightDiff;
          }
          return a.localeCompare(b, 'zh-Hans-CN');
        });

      const typeOptions = [`<option value="">${PLACEHOLDER_LABEL}</option>`].concat(
        available.map((name) => {
          const label = formatBuffLabel(name);
          const selectedAttr = currentType === name ? ' selected' : '';
          return `<option value="${escapeHtml(name)}"${selectedAttr}>${escapeHtml(label)}</option>`;
        }),
      );

      let rawValueOptions = [];
      if (currentType) {
        rawValueOptions = state.buffValueOptions.get(currentType) ?? [];
      }
      const hasValues = currentType ? rawValueOptions.length > 0 : false;
      const valueOptions = [];
      if (hasValues) {
        rawValueOptions.forEach((value) => {
          const optionLabel = formatValueLabel(currentType, value);
          const selectedAttr = Number(currentValue) === value ? ' selected' : '';
          valueOptions.push(`<option value="${value}"${selectedAttr}>${escapeHtml(optionLabel)}</option>`);
        });
      } else {
        valueOptions.push('<option value="">-</option>');
      }

      const valueSelectClass = hasValues ? 'buff-value-select' : 'buff-value-select inactive-select';
      const valueSelectDisabledAttr = hasValues ? '' : ' disabled';

      const contributionText =
        currentType && currentValue != null && state.scorer
          ? `Score ${contribution.toFixed(2)}`
          : 'Score —';
      const contributionClass = currentType && currentValue != null && state.scorer ? 'slot-score' : 'slot-score inactive';

      rows.push(`
        <div class="buff-slot">
          <div class="slot-index">#${idx + 1}</div>
          <div>
            <select class="buff-type-select styled-select" data-index="${idx}">
              ${typeOptions.join('')}
            </select>
          </div>
          <div>
            <select class="${valueSelectClass} styled-select" data-index="${idx}"${valueSelectDisabledAttr}>
              ${valueOptions.join('')}
            </select>
          </div>
          <div class="${contributionClass}">${contributionText}</div>
        </div>
      `);
    }

    container.innerHTML = rows.join('');

    container.querySelectorAll('.buff-type-select').forEach((select) => {
      select.addEventListener('change', (event) => {
        const index = Number(event.target.dataset.index);
        const value = event.target.value;
        if (!value) {
          state.buffSelections[index] = null;
          state.buffValues[index] = null;
        } else {
          state.buffSelections[index] = value;
          const options = state.buffValueOptions.get(value) ?? [];
          state.buffValues[index] = options.length > 0 ? options[0] : null;
        }
        computeContributions();
        renderBuffSlots();
        renderTotalScoreCard();
        updateComputeButtonState();
        if (state.resultId) {
          updateSuggestion();
        }
      });
    });

    container.querySelectorAll('.buff-value-select').forEach((select) => {
      select.addEventListener('change', (event) => {
        const index = Number(event.target.dataset.index);
        const selectedValue = event.target.value;
        state.buffValues[index] = selectedValue ? Number(selectedValue) : null;
        computeContributions();
        renderBuffSlots();
        renderTotalScoreCard();
        updateComputeButtonState();
        if (state.resultId) {
          updateSuggestion();
        }
      });
    });
  }

  function formatBuffLabel(buffName) {
    const label = state.buffLabels[buffName] ?? buffName;
    const weight = Number(state.weightMap[buffName] ?? 0);
    if (Math.abs(weight) < 1e-9) {
      return `（无效词条）${label}`;
    }
    return label;
  }

  function formatValueLabel(buffName, value) {
    if (state.percentBuffs.has(buffName)) {
      return `${(Number(value) / 10).toFixed(1)}%`;
    }
    return String(value);
  }

  function renderTotalScoreCard() {
    const container = elements.scoreCard;
    const totalScoreText = `${state.totalScore.toFixed(2)} / 100.00`;
    const selectedCount = state.buffSelections.filter(Boolean).length;
    const targetScore = Number(state.policySummary?.target_score ?? state.targetScore);
    const stage = state.suggestion?.stage ?? selectedCount;

    const summaryHtml = `
      <div class="score-summary">
        <div class="score-header">
          <div class="score-meter">
            <div class="score-label">当前总分</div>
            <div class="score-value">${totalScoreText}</div>
          </div>
          <div class="score-meta">
            <span class="score-target">目标分数 ${targetScore.toFixed(2)}</span>
            <span class="score-stage">已揭示 ${stage}/${state.maxSelectedTypes} 词条</span>
          </div>
        </div>
        <div class="score-suggestion">
          <div class="score-suggestion-title">强化建议</div>
          ${renderSuggestionBlock()}
        </div>
      </div>
    `;

    const warningHtml =
      state.topWeightsSum <= 0
        ? '<div class="warning">请输入至少一个大于 0 的权重以计算评分。</div>'
        : '';

    container.innerHTML = summaryHtml + warningHtml;
  }

  function renderSuggestionBlock() {
    if (!state.resultId) {
      return '<div class="empty-state">计算策略后将显示强化建议。</div>';
    }

    const rawSuggestion = state.suggestion?.suggestion ?? null;

    if (!rawSuggestion) {
      return '<div class="suggestion-box suggestion-info">暂无强化建议。</div>';
    }

    let className = 'suggestion-box';
    let text = rawSuggestion;

    if (rawSuggestion === 'Continue') {
      className += ' suggestion-continue';
      text = '建议继续';
    } else if (rawSuggestion === 'Abandon') {
      className += ' suggestion-abandon';
      text = '建议放弃';
    } else {
      className += ' suggestion-info';
    }

    return `<div class="${className}">${escapeHtml(text)}</div>`;
  }

  function updateComputeButtonState() {
    if (elements.computeButton.dataset.loading === 'true') {
      return;
    }
    elements.computeButton.disabled = !state.scorer;
  }

  function resetPolicyResult() {
    state.policySummary = null;
    state.policyError = null;
    state.resultId = null;
    state.suggestion = null;
    state.suggestionPending = false;
    renderResults();
    renderTotalScoreCard();
  }

  async function handleCompute() {
    const payload = {
      buff_weights: state.weightMap,
      target_score: state.targetScore,
      simulation_runs: state.simulationRuns,
      simulation_seed: state.simulationSeed,
      cost_weights: {
        w_echo: state.costWeights.wEcho,
        w_dkq: state.costWeights.wDkq,
        w_exp: state.costWeights.wExp,
      },
      include_user_counts: state.includeUserCounts,
    };

    elements.computeButton.dataset.loading = 'true';
    elements.computeButton.disabled = true;
    const originalText = elements.computeButton.textContent;
    elements.computeButton.textContent = '计算中…';

    try {
      const response = await window.api.computePolicy(payload);
      state.policySummary = response.summary;
      state.policyError = null;
      state.resultId = response.result_id;
      renderResults();
      state.suggestion = null;
      updateSuggestion();
    } catch (error) {
      state.policySummary = null;
      state.policyError = error.message || String(error);
      state.resultId = null;
      state.suggestion = null;
      renderResults();
      renderTotalScoreCard();
    } finally {
      elements.computeButton.dataset.loading = 'false';
      elements.computeButton.textContent = originalText;
      updateComputeButtonState();
    }
  }

  async function updateSuggestion() {
    if (!state.resultId) {
      state.suggestion = null;
      state.suggestionPending = false;
      renderTotalScoreCard();
      return;
    }
    state.suggestionPending = true;
    renderTotalScoreCard();
    const token = ++suggestionRequestToken;
    try {
      const response = await window.api.policySuggestion({
        result_id: state.resultId,
        buff_names: state.buffSelections.filter(Boolean),
        total_score: state.totalScore,
      });
      if (token === suggestionRequestToken) {
        state.suggestion = response;
        state.suggestionPending = false;
        renderTotalScoreCard();
      }
    } catch (error) {
      if (token === suggestionRequestToken) {
        state.suggestion = { suggestion: `获取建议失败：${error.message || error}` };
        state.suggestionPending = false;
        renderTotalScoreCard();
      }
    }
  }

  function renderResults() {
    const container = elements.resultsSection;
    container.innerHTML = '';

    if (state.policyError) {
      container.innerHTML = `<div class="error-message">策略计算失败：${escapeHtml(state.policyError)}</div>`;
      return;
    }

    if (!state.policySummary) {
      return;
    }

    const summary = state.policySummary;
    const lambdaStar = Number(summary.lambda_star);
    const expectedCost = Number(summary.expected_cost_per_success);
    const computeSeconds = Number(summary.compute_seconds);
    const costModel = summary.cost_model ?? {};

    const summaryCard = document.createElement('section');
    summaryCard.className = 'card';
    summaryCard.innerHTML = `
      <div class="card-title">策略计算结果</div>
      <div class="simulation-grid">
        <div class="simulation-row">
          ${renderSummaryMetric('λ*', lambdaStar.toFixed(8))}
          ${renderSummaryMetric('期望成本', Number.isFinite(expectedCost) ? expectedCost.toFixed(2) : '∞')}
        </div>
      </div>
      <div class="result-meta">DP 计算耗时 ${computeSeconds.toFixed(2)} 秒</div>
    `;

    container.appendChild(summaryCard);

    if (summary.simulation) {
      container.appendChild(renderSimulationCard(summary.simulation, costModel));
    }

    if (summary.first_upgrade_table && summary.first_upgrade_table.length > 0) {
      container.appendChild(renderFirstUpgradeCard(summary.first_upgrade_table));
    }
  }

  function renderSimulationCard(simulation, costModel) {
    const card = document.createElement('section');
    card.className = 'card simulation-card';

    const successRate = Number(simulation.success_rate ?? 0);
    const echoPerSuccess = Number(simulation.echo_per_success ?? 0);
    const dkqPerSuccess = Number(simulation.dkq_per_success ?? 0);
    const expPerSuccess = Number(simulation.exp_per_success ?? 0);
    const totalRuns = Number(simulation.total_runs ?? 0);

    const avgCost =
      Number(costModel.w_echo ?? 0) * echoPerSuccess +
      Number(costModel.w_dkq ?? 0) * dkqPerSuccess +
      Number(costModel.w_exp ?? 0) * expPerSuccess;
    const fullUpgradeRate =
      totalRuns > 0 && Array.isArray(simulation.max_slot_scores)
        ? simulation.max_slot_scores.length / totalRuns
        : 0;

    card.innerHTML = `
      <div class="card-title">蒙特卡洛模拟结果</div>
      <div class="simulation-grid">
        <div class="simulation-row">
          ${renderSimulationMetric('成功率', `${(successRate * 100).toFixed(2)}%`)}
          ${renderSimulationMetric('满强化率', `${(fullUpgradeRate * 100).toFixed(2)}%`)}
          ${renderSimulationMetric('平均成本', avgCost.toFixed(2))}
        </div>
        <div class="simulation-row">
          ${renderSimulationMetric('平均胚子消耗', echoPerSuccess.toFixed(2))}
          ${renderSimulationMetric('平均调谐器消耗', dkqPerSuccess.toFixed(2))}
          ${renderSimulationMetric('平均金密音筒消耗', expPerSuccess.toFixed(2))}
        </div>
      </div>
    `;

    return card;
  }

  function renderSimulationMetric(label, value) {
    return `
      <div class="metric">
        <span class="label">${escapeHtml(label)}</span>
        <span class="value">${escapeHtml(value)}</span>
      </div>
    `;
  }

  function renderSummaryMetric(label, value) {
    return renderSimulationMetric(label, value);
  }

  function renderFirstUpgradeCard(groups) {
    const card = document.createElement('section');
    card.className = 'card';

    const details = document.createElement('details');
    details.className = 'prob-collapse';

    const summary = document.createElement('summary');
    summary.className = 'prob-collapse-summary';
    summary.textContent = '首条检视 (继续条件表)';
    details.appendChild(summary);

    const body = document.createElement('div');
    body.className = 'prob-collapse-body';

    const buffSlotProb = 1 / state.buffTypes.length;
    let totalContinueProb = 0;

    groups.forEach((group) => {
      const label = state.buffLabels[group.buff_name] ?? group.buff_name;
      const options = [...(group.options ?? [])].sort((a, b) => Number(b.raw_value) - Number(a.raw_value));
      if (options.length === 0) {
        return;
      }

      let cumulative = 0;
      const rows = options
        .map((option) => {
          const singleProb = Number(option.probability ?? 0) * buffSlotProb;
          cumulative += singleProb;
          return `
            <tr>
              <td>${escapeHtml(formatValueLabel(group.buff_name, Number(option.raw_value)))}</td>
              <td>${(singleProb * 100).toFixed(2)}%</td>
              <td>${(cumulative * 100).toFixed(2)}%</td>
            </tr>
          `;
        })
        .join('');

      totalContinueProb += cumulative;

      const section = document.createElement('div');
      section.className = 'prob-group';
      section.innerHTML = `
        <div class="prob-header">
          <span>${escapeHtml(label)}</span>
          <span>累计概率 ${(cumulative * 100).toFixed(2)}%</span>
        </div>
        <table class="prob-table">
          <thead>
            <tr>
              <th>数值</th>
              <th>概率(%)</th>
              <th>累计概率(%)</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      `;
      body.appendChild(section);
    });

    const totalSummary = document.createElement('div');
    totalSummary.style.marginTop = '0.75rem';
    totalSummary.style.fontWeight = '600';
    totalSummary.textContent = `总继续概率：${(totalContinueProb * 100).toFixed(2)}%`;
    body.appendChild(totalSummary);

    details.appendChild(body);
    card.appendChild(details);

    return card;
  }

  document.addEventListener('DOMContentLoaded', init);
})();
