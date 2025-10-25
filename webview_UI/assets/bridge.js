(function () {
  let resolveReady;
  let rejectReady;
  const readyPromise = new Promise((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
  });

  function markReady(api) {
    if (!api) {
      rejectReady(new Error('PyWebview API is unavailable.'));
      return;
    }
    resolveReady(api);
  }

  if (window.pywebview && window.pywebview.api) {
    markReady(window.pywebview.api);
  } else {
    window.addEventListener('pywebviewready', () => {
      markReady(window.pywebview && window.pywebview.api);
    });
  }

  async function getPyApi() {
    const api = await readyPromise;
    if (!api) {
      throw new Error('PyWebview API is unavailable.');
    }
    return api;
  }

  async function callPyApi(name, payload) {
    const api = await getPyApi();
    if (typeof api[name] !== 'function') {
      throw new Error(`PyWebview API method '${name}' is not available`);
    }
    if (payload === undefined) {
      return api[name]();
    }
    return api[name](payload ?? {});
  }

  const bridge = {
    platform: null,
    async bootstrap() {
      return callPyApi('bootstrap');
    },
    async setExpRefundRatio(value) {
      return callPyApi('set_exp_refund_ratio', { value });
    },
    async computePolicy(payload) {
      return callPyApi('compute_policy', payload);
    },
    async policySuggestion(payload) {
      return callPyApi('policy_suggestion', payload);
    },
  };

  window.api = bridge;

  (async () => {
    try {
      const pyApi = await getPyApi();
      if (pyApi && typeof pyApi.get_platform === 'function') {
        bridge.platform = await pyApi.get_platform();
        return;
      }
    } catch {
      /* ignore */
    }
    bridge.platform = navigator.platform.startsWith('Win') ? 'win32' : 'unknown';
  })();
})();
