/* global API, Toast, Modal, App */

/**
 * Config Studio (schema-driven config editor)
 *
 * This file defines `window.ConfigStudio` and is loaded before app.js.
 * It only relies on API/Toast/Modal at call time (after app.js initializes).
 */

(function () {
    const ConfigStudio = {
        _inited: false,
        state: {
            currentOverride: '',
            editTarget: 'base', // base|override
            view: 'overview',
            schema: null,
            base: {},
            override: {},
            effective: {},
            effective_processed: {},
            meta: {},
            pendingOps: new Map(), // pathKey -> op
            selectedModels: new Set(),
            activeModel: null,
            modelFilter: '',
            selectedPreset: null,
        },

        ensureInit() {
            if (this._inited) return;
            this._inited = true;

            const bind = (id, event, handler) => {
                const el = document.getElementById(id);
                if (el) el.addEventListener(event, handler);
            };

            bind('cfg-target-override', 'click', () => this.setEditTarget('override'));
            bind('cfg-target-base', 'click', () => this.setEditTarget('base'));
            bind('btn-config-save', 'click', () => this.save());
            bind('btn-config-validate', 'click', () => this.validate());
            bind('btn-config-yaml', 'click', () => this.showYamlPreview());
        },

        _getEls() {
            return {
                nav: document.getElementById('config-studio-nav'),
                main: document.getElementById('config-studio-main'),
                statusDot: document.getElementById('cfg-status-dot'),
                statusText: document.getElementById('cfg-status-text'),
                dirty: document.getElementById('cfg-dirty-indicator'),
                btnSave: document.getElementById('btn-config-save'),
                btnTargetOverride: document.getElementById('cfg-target-override'),
                btnTargetBase: document.getElementById('cfg-target-base'),
            };
        },

        _escape(s) {
            return String(s || '')
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/\"/g, '&quot;')
                .replace(/'/g, '&#039;');
        },

        setStatus(kind, text) {
            const { statusDot, statusText } = this._getEls();
            if (statusText) statusText.textContent = text || '';
            if (statusDot) {
                statusDot.classList.remove('connected', 'disconnected');
                statusDot.classList.add(kind === 'ok' ? 'connected' : 'disconnected');
            }
        },

        setDirty(isDirty) {
            const { dirty, btnSave } = this._getEls();
            const canWrite = (App?.capabilities?.can_write_configs !== false);
            const hasOverride = !!this.state.currentOverride;
            const targetOk = this.state.editTarget === 'base' || hasOverride;

            if (dirty) dirty.textContent = isDirty ? 'Unsaved changes' : '';
            if (btnSave) btnSave.disabled = !canWrite || !targetOk || !isDirty;
        },

        setEditTarget(target) {
            if (target !== 'base' && target !== 'override') return;
            this.state.editTarget = target;
            this.state.pendingOps.clear();
            this.setDirty(false);
            this._updateTargetButtons();
            this.render();
        },

        _updateTargetButtons() {
            const { btnTargetOverride, btnTargetBase } = this._getEls();
            if (btnTargetOverride) btnTargetOverride.classList.toggle('active', this.state.editTarget === 'override');
            if (btnTargetBase) btnTargetBase.classList.toggle('active', this.state.editTarget === 'base');
        },

        async load(currentOverride) {
            this.ensureInit();
            this.state.currentOverride = currentOverride || '';
            this.state.pendingOps.clear();
            this.setDirty(false);

            // Default: edit override if one is selected, else base.
            this.state.editTarget = this.state.currentOverride ? 'override' : 'base';
            this._updateTargetButtons();

            this.setStatus('loading', 'Loading...');
            const url = this.state.currentOverride
                ? `/api/config/studio?override=${encodeURIComponent(this.state.currentOverride)}`
                : '/api/config/studio';

            try {
                const data = await API.get(url);
                this.state.schema = data.schema || null;
                this.state.base = data.base || {};
                this.state.override = data.override || {};
                this.state.effective = data.effective || {};
                this.state.effective_processed = data.effective_processed || {};
                this.state.meta = data.meta || {};

                // Keep selections stable across reloads (best-effort)
                const modelNames = new Set((this.state.meta.models || []).map(String));
                const nextSelected = new Set();
                for (const m of this.state.selectedModels) {
                    if (modelNames.has(m)) nextSelected.add(m);
                }
                this.state.selectedModels = nextSelected;
                if (this.state.activeModel && !modelNames.has(this.state.activeModel)) {
                    this.state.activeModel = null;
                }

                this.setStatus('ok', 'Loaded');
                this.renderNav();
                this.render();
            } catch (e) {
                console.error('Config Studio load failed:', e);
                this.setStatus('error', `Failed to load: ${e.message}`);
                const { main } = this._getEls();
                if (main) main.innerHTML = `<p class="empty-message">Failed to load Config Studio: ${this._escape(e.message)}</p>`;
            }
        },

        renderNav() {
            const { nav } = this._getEls();
            if (!nav) return;

            const schema = this.state.schema || {};
            const groups = Array.isArray(schema.groups) ? schema.groups : [];
            const topGroups = groups.filter(g => g && (g.kind || ['overview', 'general', 'defaults', 'presets', 'models', 'bulk_apply'].includes(g.id)));

            nav.innerHTML = '';
            for (const g of topGroups) {
                const id = g.kind || g.id;
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.textContent = g.title || id;
                btn.classList.toggle('active', this.state.view === id);
                btn.addEventListener('click', () => {
                    this.state.view = id;
                    nav.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    this.render();
                });
                nav.appendChild(btn);
            }
        },

        render() {
            const { main } = this._getEls();
            if (!main) return;

            this._updateTargetButtons();

            const view = this.state.view || 'overview';
            if (view === 'overview') {
                main.innerHTML = this._renderOverview();
                return;
            }

            if (view === 'models') {
                main.innerHTML = this._renderModels();
                this._bindModelsUI();
                return;
            }

            if (view === 'presets') {
                main.innerHTML = this._renderPresets();
                this._bindPresetsUI();
                return;
            }

            if (view === 'bulk_apply') {
                main.innerHTML = this._renderBulkApply();
                this._bindBulkApplyUI();
                return;
            }

            main.innerHTML = this._renderGroupFields(view);
            this._bindFieldInputs(main);
        },

        _renderOverview() {
            const modelsCount = (this.state.meta.models || []).length;
            const presetsCount = (this.state.meta.sampling_presets || []).length;
            const targetLabel = this.state.editTarget === 'override'
                ? (this.state.currentOverride ? `Override: ${this._escape(this.state.currentOverride)}` : 'Override: (none selected)')
                : 'Base config';

            return `
                <div class="studio-section-title">Overview</div>
                <div class="field-grid">
                    <div class="field-card">
                        <div class="field-label">Editing</div>
                        <div class="field-help">${targetLabel}</div>
                    </div>
                    <div class="field-card">
                        <div class="field-label">Models</div>
                        <div class="field-help">${modelsCount} configured</div>
                    </div>
                    <div class="field-card">
                        <div class="field-label">Sampling presets</div>
                        <div class="field-help">${presetsCount} preset(s) in base config</div>
                    </div>
                </div>
            `;
        },

        _schemaGroupById(idOrKind) {
            const schema = this.state.schema || {};
            const groups = Array.isArray(schema.groups) ? schema.groups : [];
            return groups.find(g => (g.kind || g.id) === idOrKind) || null;
        },

        _renderGroupFields(groupId) {
            const group = this._schemaGroupById(groupId);
            if (!group) return `<p class="empty-message">Unknown view: ${this._escape(groupId)}</p>`;
            const fields = Array.isArray(group.fields) ? group.fields : [];

            return `
                <div class="studio-section-title">${this._escape(group.title || groupId)}</div>
                <div class="field-grid">
                    ${fields.map(f => this._renderFieldCard(f)).join('')}
                </div>
            `;
        },

        _pathKey(path) {
            return (path || []).map(p => String(p)).join('|');
        },

        _hasPath(obj, path) {
            let cur = obj;
            for (const key of (path || [])) {
                if (cur == null) return false;
                if (typeof cur !== 'object') return false;
                if (!(key in cur)) return false;
                cur = cur[key];
            }
            return true;
        },

        _getPath(obj, path) {
            let cur = obj;
            for (const key of (path || [])) {
                if (cur == null) return undefined;
                cur = cur[key];
            }
            return cur;
        },

        _fieldValueForInput(field, fullPath) {
            const baseVal = this._getPath(this.state.base, fullPath);
            const overrideVal = this._getPath(this.state.override, fullPath);
            const effectiveVal = this._getPath(this.state.effective, fullPath);

            if (this.state.editTarget === 'base') return baseVal;
            return (this._hasPath(this.state.override, fullPath) ? overrideVal : effectiveVal);
        },

        _sourceBadge(field, fullPath) {
            const hasOvr = this._hasPath(this.state.override, fullPath);
            const hasBase = this._hasPath(this.state.base, fullPath);
            const label = hasOvr ? 'Override' : (hasBase ? 'Base' : 'Unset');
            const cls = hasOvr ? 'override' : (hasBase ? 'base' : 'unset');

            const overridden = this.state.editTarget === 'base' && this.state.currentOverride && hasOvr;
            const overriddenBadge = overridden ? `<span class="badge unset" title="Overridden by current override">Overridden</span>` : '';

            return `<span class="badge ${cls}">${label}</span>${overriddenBadge}`;
        },

        _renderFieldCard(field, opts = {}) {
            const prefix = opts.prefix || [];
            const fullPath = [...prefix, ...(field.path || [])];
            const inputId = `cfg-field-${this._pathKey(fullPath).replace(/\|/g, '-')}`;

            const canReset = this.state.editTarget === 'override' && this._hasPath(this.state.override, fullPath);
            const help = field.help ? `<div class="field-help">${this._escape(field.help)}</div>` : '';

            const value = this._fieldValueForInput(field, fullPath);
            const ui = field.ui || 'text';

            let inputHtml = '';
            if (ui === 'toggle' && field.type === 'bool') {
                const checked = !!value;
                inputHtml = `<label class="checkbox-label"><input type="checkbox" id="${inputId}" ${checked ? 'checked' : ''}> ${this._escape(field.label || field.id)}</label>`;
            } else if (ui === 'select' && Array.isArray(field.enum_values)) {
                const v = (value == null) ? '' : String(value);
                inputHtml = `
                    <select id="${inputId}">
                        <option value="">(inherit)</option>
                        ${field.enum_values.map(opt => `<option value="${this._escape(String(opt))}" ${String(opt) === v ? 'selected' : ''}>${this._escape(String(opt))}</option>`).join('')}
                    </select>
                `;
            } else if (ui === 'number') {
                const v = (value == null || value === '-') ? '' : String(value);
                const min = (field.min != null) ? `min="${field.min}"` : '';
                const max = (field.max != null) ? `max="${field.max}"` : '';
                const step = (field.step != null) ? `step="${field.step}"` : '';
                inputHtml = `<input type="number" id="${inputId}" value="${this._escape(v)}" ${min} ${max} ${step}>`;
            } else if (ui === 'tags' && field.type === 'list[str]') {
                const v = Array.isArray(value) ? value.join(', ') : '';
                inputHtml = `<input type="text" id="${inputId}" value="${this._escape(v)}" placeholder="${this._escape(field.placeholder || 'Comma-separated')}">`;
            } else {
                const v = (value == null || value === '-') ? '' : String(value);
                inputHtml = `<input type="text" id="${inputId}" value="${this._escape(v)}" placeholder="${this._escape(field.placeholder || '')}">`;
            }

            return `
                <div class="field-card" data-field-path="${this._escape(JSON.stringify(fullPath))}" data-field-meta="${this._escape(JSON.stringify(field))}">
                    <div class="field-header">
                        <div class="field-label">${this._escape(field.label || field.id)}</div>
                        <div style="display:flex; gap:6px; align-items:center;">
                            ${this._sourceBadge(field, fullPath)}
                            ${canReset ? `<button type="button" class="icon-btn" data-action="reset" title="Reset to base">Reset</button>` : ''}
                        </div>
                    </div>
                    ${inputHtml}
                    ${help}
                </div>
            `;
        },

        _bindFieldInputs(rootEl) {
            const cards = rootEl.querySelectorAll('.field-card');
            cards.forEach(card => {
                const metaRaw = card.getAttribute('data-field-meta') || '{}';
                const pathRaw = card.getAttribute('data-field-path') || '[]';
                let field = {};
                let fullPath = [];
                try { field = JSON.parse(metaRaw); } catch {}
                try { fullPath = JSON.parse(pathRaw); } catch {}

                const input = card.querySelector('input, select, textarea');
                if (input) {
                    input.addEventListener('change', () => {
                        this._queueFieldSet(field, fullPath, input);
                    });
                }
                const resetBtn = card.querySelector('button[data-action="reset"]');
                if (resetBtn) {
                    resetBtn.addEventListener('click', () => {
                        this._queueDelete(fullPath);
                        const fallbackVal = this._getPath(this.state.base, fullPath);
                        if (input) {
                            if (input.type === 'checkbox') input.checked = !!fallbackVal;
                            else input.value = (fallbackVal == null) ? '' : String(fallbackVal);
                        }
                    });
                }
            });
        },

        _queueDelete(fullPath) {
            const key = this._pathKey(fullPath);
            this.state.pendingOps.set(key, { op: 'delete', path: fullPath });
            this.setDirty(true);
        },

        _queueFieldSet(field, fullPath, inputEl) {
            const ui = field.ui || 'text';
            const type = field.type || 'str';

            let value = null;
            try {
                if (ui === 'toggle' && type === 'bool') {
                    value = !!inputEl.checked;
                } else if (ui === 'number') {
                    const raw = String(inputEl.value || '').trim();
                    if (!raw) {
                        this._queueDelete(fullPath);
                        return;
                    }
                    value = (type === 'float') ? parseFloat(raw) : parseInt(raw, 10);
                    if (Number.isNaN(value)) throw new Error('Invalid number');
                } else if (ui === 'select' && Array.isArray(field.enum_values)) {
                    const raw = String(inputEl.value || '');
                    if (!raw) {
                        this._queueDelete(fullPath);
                        return;
                    }
                    value = raw;
                } else if (ui === 'tags' && type === 'list[str]') {
                    const raw = String(inputEl.value || '').trim();
                    if (!raw) {
                        this._queueDelete(fullPath);
                        return;
                    }
                    value = raw.split(',').map(s => s.trim()).filter(Boolean);
                } else if (type === 'int_or_auto') {
                    const raw = String(inputEl.value || '').trim();
                    if (!raw) {
                        this._queueDelete(fullPath);
                        return;
                    }
                    if (raw.toLowerCase() === 'auto') value = 'auto';
                    else {
                        const n = parseInt(raw, 10);
                        if (Number.isNaN(n)) throw new Error('Invalid int');
                        value = n;
                    }
                } else {
                    const raw = String(inputEl.value || '').trim();
                    if (!raw) {
                        this._queueDelete(fullPath);
                        return;
                    }
                    value = raw;
                }
            } catch (e) {
                Toast.error(e.message || 'Invalid input');
                return;
            }

            const key = this._pathKey(fullPath);
            this.state.pendingOps.set(key, { op: 'set', path: fullPath, value });
            this.setDirty(true);
        },

        async save() {
            const hasOps = this.state.pendingOps.size > 0;
            if (!hasOps) return;

            const canWrite = (App?.capabilities?.can_write_configs !== false);
            if (!canWrite) {
                Toast.error('Config writes are disabled in this mode');
                return;
            }

            if (this.state.editTarget === 'override' && !this.state.currentOverride) {
                Toast.error('Select an override (top-right) or switch to Base');
                return;
            }

            const ops = Array.from(this.state.pendingOps.values());
            const target = (this.state.editTarget === 'override')
                ? { kind: 'override', name: this.state.currentOverride }
                : { kind: 'base' };

            try {
                await API.post('/api/config/studio/patch', {
                    target,
                    ops,
                    context_override: this.state.currentOverride || null,
                });
                Toast.success('Saved');
                await this.load(this.state.currentOverride);
            } catch (e) {
                Toast.error(`Save failed: ${e.message}`);
            }
        },

        async validate() {
            try {
                const url = this.state.currentOverride
                    ? `/api/config/studio/validate?override=${encodeURIComponent(this.state.currentOverride)}`
                    : '/api/config/studio/validate';
                await API.post(url, {});
                Toast.success('Config validated');
            } catch (e) {
                Toast.error(`Validation failed: ${e.message}`);
            }
        },

        async showYamlPreview() {
            const loadBase = async () => {
                const data = await API.get('/api/config');
                return data.content || '';
            };
            const loadOverride = async () => {
                if (!this.state.currentOverride) return '';
                const data = await API.get(`/api/config/overrides/${encodeURIComponent(this.state.currentOverride)}`);
                return data.content || '';
            };
            const loadEffective = async () => {
                const url = this.state.currentOverride
                    ? `/api/config/effective?override=${encodeURIComponent(this.state.currentOverride)}`
                    : '/api/config/effective';
                const data = await API.get(url);
                return data.yaml || '';
            };

            try {
                const [baseYaml, overrideYaml, effectiveYaml] = await Promise.all([
                    loadBase(),
                    loadOverride(),
                    loadEffective(),
                ]);

                const tabs = [
                    { id: 'base', title: 'Base', content: baseYaml },
                    { id: 'override', title: 'Override', content: overrideYaml || '(no override selected)' },
                    { id: 'effective', title: 'Effective', content: effectiveYaml },
                ];

                const tabBtns = tabs.map((t, i) => `<button class="tab-btn ${i === 0 ? 'active' : ''}" data-tab="yaml-${t.id}">${t.title}</button>`).join('');
                const tabBodies = tabs.map((t, i) => `
                    <div id="yaml-${t.id}" class="tab-content ${i === 0 ? 'active' : ''}">
                        <div class="editor-container">
                            <pre class="code-display" style="max-height: 60vh; overflow:auto; white-space: pre;">${this._escape(t.content)}</pre>
                        </div>
                    </div>
                `).join('');

                Modal.show('YAML Preview', `
                    <div class="section-tabs">${tabBtns}</div>
                    ${tabBodies}
                `, `
                    <button class="btn btn-secondary" onclick="Modal.hide()">Close</button>
                `);

                try { App.setupTabs(); } catch {}
            } catch (e) {
                Toast.error(`Failed to load YAML: ${e.message}`);
            }
        },

        // -------- Models --------
        _renderModels() {
            const models = this.state.meta.models || [];
            const filter = this.state.modelFilter || '';
            const shown = models.filter(m => !filter || String(m).toLowerCase().includes(filter.toLowerCase()));

            const active = this.state.activeModel || (shown[0] || null);
            this.state.activeModel = active;

            const modelRows = shown.map(m => {
                const name = String(m);
                const checked = this.state.selectedModels.has(name);
                const isActive = active === name;
                return `
                    <div class="model-row ${isActive ? 'selected' : ''}" data-model="${this._escape(name)}">
                        <input type="checkbox" ${checked ? 'checked' : ''} data-model-check="${this._escape(name)}">
                        <div class="model-name">${this._escape(name)}</div>
                        <div class="model-sub">${checked ? 'selected' : ''}</div>
                    </div>
                `;
            }).join('');

            const editor = active ? this._renderModelEditor(active) : '<p class="empty-message">No models found.</p>';

            return `
                <div class="studio-section-title">Models</div>
                <div class="studio-split">
                    <div class="model-list">
                        <div class="model-list-header">
                            <input type="text" id="cfg-model-filter" class="search-input" placeholder="Search models..." value="${this._escape(filter)}">
                            <div style="display:flex; gap:8px;">
                                <button type="button" class="btn btn-secondary btn-sm" id="cfg-select-all-filtered">Select filtered</button>
                                <button type="button" class="btn btn-secondary btn-sm" id="cfg-clear-selection">Clear</button>
                            </div>
                            <div class="muted">${shown.length}/${models.length} shown</div>
                        </div>
                        <div class="model-list-body" id="cfg-model-list">
                            ${modelRows || '<p class="empty-message" style="padding:0.75rem;">No models match.</p>'}
                        </div>
                    </div>
                    <div id="cfg-model-editor">
                        ${editor}
                    </div>
                </div>
            `;
        },

        _renderModelEditor(modelName) {
            const schema = this.state.schema || {};
            const groups = Array.isArray(schema.groups) ? schema.groups : [];

            const modelGroups = groups.filter(g => g && g.scope === 'model' && Array.isArray(g.fields));
            const processedModel = (((this.state.effective_processed || {}).models || {})[modelName]) || {};
            const cmdStr = processedModel.generated_cmd_str ? String(processedModel.generated_cmd_str) : '';

            const prefix = ['models', modelName];
            const blocks = modelGroups.map(g => `
                <div class="studio-section-title" style="margin-top: 1rem;">${this._escape(g.title || g.id)}</div>
                <div class="field-grid">
                    ${(g.fields || []).map(f => this._renderFieldCard(f, { prefix })).join('')}
                </div>
            `).join('');

            const presetOptions = (this.state.meta.sampling_presets || []).map(p => `<option value="${this._escape(String(p))}">${this._escape(String(p))}</option>`).join('');
            const selectedCount = this.state.selectedModels.size;

            return `
                <div class="field-card">
                    <div class="field-header">
                        <div class="field-label">${this._escape(modelName)}</div>
                        <div style="display:flex; gap:8px; align-items:center;">
                            <select id="cfg-preset-select">
                                <option value="">Apply sampling preset...</option>
                                ${presetOptions}
                            </select>
                            <button type="button" class="btn btn-secondary btn-sm" id="cfg-apply-preset-to-model">Apply</button>
                        </div>
                    </div>
                    <div class="field-help">
                        Effective values reflect base + current override. Use Reset (override mode) to fall back to base.
                    </div>
                </div>

                <div class="field-card" style="margin-top: 1rem;">
                    <div class="field-header">
                        <div class="field-label">Copy settings</div>
                        <div class="field-source-badge">${selectedCount} selected</div>
                    </div>
                    <div class="field-help">
                        Copy <code>cmd</code> (excluding <code>bin</code>/<code>model</code>/<code>port</code>) and/or <code>sampling</code>
                        from this model to the currently selected models.
                    </div>
                    <div style="display:flex; gap:12px; align-items:center; flex-wrap: wrap; margin-top: 0.5rem;">
                        <label class="checkbox-label"><input type="checkbox" id="cfg-copy-cmd" checked> Cmd</label>
                        <label class="checkbox-label"><input type="checkbox" id="cfg-copy-sampling" checked> Sampling</label>
                        <button type="button" class="btn btn-secondary btn-sm" id="cfg-copy-apply">Copy to selected</button>
                    </div>
                </div>

                ${blocks}

                <div class="studio-section-title" style="margin-top: 1rem;">Generated command</div>
                <div class="field-card">
                    <pre class="code-display" style="max-height: 240px; overflow:auto; white-space: pre-wrap;">${this._escape(cmdStr || '(not available)')}</pre>
                </div>
            `;
        },

        _bindModelsUI() {
            const filterEl = document.getElementById('cfg-model-filter');
            if (filterEl) {
                filterEl.addEventListener('input', (e) => {
                    this.state.modelFilter = e.target.value || '';
                    this.render();
                });
            }

            const list = document.getElementById('cfg-model-list');
            if (list) {
                list.querySelectorAll('.model-row').forEach(row => {
                    const model = row.getAttribute('data-model');
                    row.addEventListener('click', (e) => {
                        if (e.target && e.target.matches('input[type=\"checkbox\"]')) return;
                        this.state.activeModel = model;
                        this.render();
                    });
                });
                list.querySelectorAll('input[data-model-check]').forEach(cb => {
                    cb.addEventListener('change', () => {
                        const model = cb.getAttribute('data-model-check');
                        if (cb.checked) this.state.selectedModels.add(model);
                        else this.state.selectedModels.delete(model);
                        this.render();
                    });
                });
            }

            const selectAllBtn = document.getElementById('cfg-select-all-filtered');
            if (selectAllBtn) {
                selectAllBtn.addEventListener('click', () => {
                    const models = (this.state.meta.models || []).map(String);
                    const filter = (this.state.modelFilter || '').toLowerCase();
                    models.forEach(m => {
                        if (!filter || m.toLowerCase().includes(filter)) this.state.selectedModels.add(m);
                    });
                    this.render();
                });
            }
            const clearBtn = document.getElementById('cfg-clear-selection');
            if (clearBtn) {
                clearBtn.addEventListener('click', () => {
                    this.state.selectedModels.clear();
                    this.render();
                });
            }

            const editor = document.getElementById('cfg-model-editor');
            if (editor) {
                this._bindFieldInputs(editor);

                const applyPresetBtn = document.getElementById('cfg-apply-preset-to-model');
                if (applyPresetBtn) {
                    applyPresetBtn.addEventListener('click', async () => {
                        const modelName = this.state.activeModel;
                        const preset = document.getElementById('cfg-preset-select')?.value || '';
                        if (!modelName || !preset) return;

                        const target = (this.state.editTarget === 'override')
                            ? { kind: 'override', name: this.state.currentOverride }
                            : { kind: 'base' };

                        try {
                            await API.post('/api/config/studio/presets/apply', {
                                preset_name: preset,
                                target,
                                models: [modelName],
                                context_override: this.state.currentOverride || null,
                            });
                            Toast.success('Preset applied');
                            await this.load(this.state.currentOverride);
                            this.state.view = 'models';
                            this.renderNav();
                            this.render();
                        } catch (e) {
                            Toast.error(`Preset apply failed: ${e.message}`);
                        }
                    });
                }

                const copyBtn = document.getElementById('cfg-copy-apply');
                if (copyBtn) {
                    copyBtn.addEventListener('click', async () => {
                        const sourceModel = this.state.activeModel;
                        const targetsRaw = Array.from(this.state.selectedModels).map(String);
                        const targets = targetsRaw.filter((m) => m && m !== sourceModel);
                        if (!sourceModel) return;
                        if (targets.length === 0) {
                            Toast.error('Select one or more target models (checkboxes on the left)');
                            return;
                        }

                        const includeCmd = !!document.getElementById('cfg-copy-cmd')?.checked;
                        const includeSampling = !!document.getElementById('cfg-copy-sampling')?.checked;
                        if (!includeCmd && !includeSampling) {
                            Toast.error('Select what to copy (Cmd and/or Sampling)');
                            return;
                        }

                        const target = (this.state.editTarget === 'override')
                            ? { kind: 'override', name: this.state.currentOverride }
                            : { kind: 'base' };
                        if (target.kind === 'override' && !target.name) {
                            Toast.error('Select an override (top-right) or switch to Base');
                            return;
                        }

                        const effModels = ((this.state.effective || {}).models || {});
                        const srcCfg = effModels[sourceModel] || {};
                        const srcCmd = srcCfg.cmd;
                        const srcSampling = srcCfg.sampling;

                        const clone = (v) => {
                            try { return JSON.parse(JSON.stringify(v)); } catch { return v; }
                        };

                        const ops = [];
                        if (includeCmd) {
                            if (!srcCmd || typeof srcCmd !== 'object') {
                                Toast.error('Source model has no cmd block');
                                return;
                            }
                            const exclude = new Set(['bin', 'model', 'port']);
                            for (const t of targets) {
                                for (const [k, v] of Object.entries(srcCmd)) {
                                    if (exclude.has(k)) continue;
                                    ops.push({ op: 'set', path: ['models', t, 'cmd', k], value: clone(v) });
                                }
                            }
                        }
                        if (includeSampling) {
                            if (!srcSampling || typeof srcSampling !== 'object') {
                                Toast.error('Source model has no sampling block');
                                return;
                            }
                            for (const t of targets) {
                                ops.push({ op: 'set', path: ['models', t, 'sampling'], value: clone(srcSampling) });
                            }
                        }

                        if (ops.length === 0) return;

                        try {
                            await API.post('/api/config/studio/patch', {
                                target,
                                ops,
                                context_override: this.state.currentOverride || null,
                            });
                            Toast.success('Copied');
                            await this.load(this.state.currentOverride);
                            this.state.view = 'models';
                            this.renderNav();
                            this.render();
                        } catch (e) {
                            Toast.error(`Copy failed: ${e.message}`);
                        }
                    });
                }
            }
        },

        // -------- Presets --------
        _renderPresets() {
            const presets = (this.state.meta.sampling_presets || []).map(String);
            const selectedPreset = this.state.selectedPreset || (presets[0] || null);
            this.state.selectedPreset = selectedPreset;

            const listHtml = presets.map(p => `
                <div class="list-item ${p === selectedPreset ? 'active' : ''}" data-preset="${this._escape(p)}">
                    <span>${this._escape(p)}</span>
                </div>
            `).join('');

            const presetObj = (selectedPreset && this.state.base[selectedPreset]) ? this.state.base[selectedPreset] : {};
            const keys = Object.keys(presetObj || {});

            const rows = keys.length
                ? keys.map(k => `
                    <div class="sweep-dim-row" style="grid-template-columns: 1fr 1fr auto;">
                        <input type="text" value="${this._escape(k)}" disabled>
                        <input type="text" class="preset-val" data-key="${this._escape(k)}" value="${this._escape(presetObj[k])}">
                        <button type="button" class="btn btn-secondary btn-sm preset-del" data-key="${this._escape(k)}">Clear</button>
                    </div>
                `).join('')
                : '<p class="empty-message">No keys in this preset.</p>';

            return `
                <div class="studio-section-title">Sampling Presets</div>
                <div class="studio-split">
                    <div class="model-list">
                        <div class="model-list-header">
                            <div class="field-label">Presets</div>
                            <div class="muted">Presets live in the base config.</div>
                        </div>
                        <div class="model-list-body" id="preset-list">
                            ${listHtml || '<p class="empty-message" style="padding:0.75rem;">No presets found.</p>'}
                        </div>
                    </div>
                    <div>
                        <div class="field-card">
                            <div class="field-header">
                                <div class="field-label">${this._escape(selectedPreset || 'Select a preset')}</div>
                                <div style="display:flex; gap:8px;">
                                    <button type="button" class="btn btn-secondary" id="btn-preset-apply">Apply to selected models</button>
                                    <button type="button" class="btn btn-primary" id="btn-preset-save">Save preset</button>
                                </div>
                            </div>
                            <div id="preset-editor">
                                ${rows}
                                <div class="sweep-actions">
                                    <button type="button" class="btn btn-secondary" id="btn-preset-add-key">+ Add key</button>
                                </div>
                                <div class="field-help">Apply sets the preset for selected models (base uses YAML alias, override copies values).</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        },

        _bindPresetsUI() {
            const list = document.getElementById('preset-list');
            if (list) {
                list.querySelectorAll('.list-item').forEach(item => {
                    item.addEventListener('click', () => {
                        this.state.selectedPreset = item.getAttribute('data-preset');
                        this.render();
                    });
                });
            }

            document.querySelectorAll('.preset-del').forEach(btn => {
                btn.addEventListener('click', () => {
                    const key = btn.getAttribute('data-key');
                    const val = document.querySelector(`.preset-val[data-key="${CSS.escape(key)}"]`);
                    if (val) val.value = '';
                });
            });

            const addKeyBtn = document.getElementById('btn-preset-add-key');
            if (addKeyBtn) {
                addKeyBtn.addEventListener('click', () => {
                    const editor = document.getElementById('preset-editor');
                    if (!editor) return;
                    const row = document.createElement('div');
                    row.className = 'sweep-dim-row';
                    row.style.gridTemplateColumns = '1fr 1fr auto';
                    row.innerHTML = `
                        <input type="text" class="preset-key-new" placeholder="key">
                        <input type="text" class="preset-val-new" placeholder="value">
                        <button type="button" class="btn btn-secondary btn-sm preset-del-new">Remove</button>
                    `;
                    editor.insertBefore(row, editor.querySelector('.sweep-actions'));
                    row.querySelector('.preset-del-new').addEventListener('click', () => row.remove());
                });
            }

            const saveBtn = document.getElementById('btn-preset-save');
            if (saveBtn) {
                saveBtn.addEventListener('click', async () => {
                    const presetName = this.state.selectedPreset;
                    if (!presetName) return;

                    const values = {};
                    document.querySelectorAll('#preset-editor .preset-val').forEach(inp => {
                        const k = inp.getAttribute('data-key');
                        const v = String(inp.value || '').trim();
                        values[k] = v === '' ? null : v;
                    });
                    document.querySelectorAll('#preset-editor .preset-key-new').forEach(inp => {
                        const k = String(inp.value || '').trim();
                        const vEl = inp.parentElement?.querySelector('.preset-val-new');
                        const v = vEl ? String(vEl.value || '').trim() : '';
                        if (k) values[k] = v;
                    });

                    try {
                        await API.post('/api/config/studio/presets/update', {
                            preset_name: presetName,
                            values,
                            context_override: this.state.currentOverride || null,
                        });
                        Toast.success('Preset saved');
                        await this.load(this.state.currentOverride);
                        this.state.view = 'presets';
                        this.renderNav();
                        this.render();
                    } catch (e) {
                        Toast.error(`Preset save failed: ${e.message}`);
                    }
                });
            }

            const applyBtn = document.getElementById('btn-preset-apply');
            if (applyBtn) {
                applyBtn.addEventListener('click', async () => {
                    const presetName = this.state.selectedPreset;
                    if (!presetName) return;
                    const selected = Array.from(this.state.selectedModels);
                    if (selected.length === 0) {
                        Toast.error('Select one or more models first (Config Studio - Models)');
                        return;
                    }
                    const target = (this.state.editTarget === 'override')
                        ? { kind: 'override', name: this.state.currentOverride }
                        : { kind: 'base' };
                    try {
                        await API.post('/api/config/studio/presets/apply', {
                            preset_name: presetName,
                            target,
                            models: selected,
                            context_override: this.state.currentOverride || null,
                        });
                        Toast.success('Preset applied');
                        await this.load(this.state.currentOverride);
                    } catch (e) {
                        Toast.error(`Preset apply failed: ${e.message}`);
                    }
                });
            }
        },

        // -------- Bulk apply --------
        getSweepableFields() {
            const schema = this.state.schema || {};
            const groups = Array.isArray(schema.groups) ? schema.groups : [];
            const out = [];
            for (const g of groups) {
                if (!g || g.scope !== 'model') continue;
                if (!Array.isArray(g.fields)) continue;
                for (const f of g.fields) {
                    if (!f || !Array.isArray(f.path) || f.path.length < 2) continue;
                    const [section, key] = f.path;
                    if (section !== 'cmd' && section !== 'sampling') continue;
                    out.push({
                        group: g.title || g.id,
                        section,
                        key,
                        label: f.label || `${section}.${key}`,
                        type: f.type || 'str',
                        ui: f.ui || 'text',
                        enum_values: f.enum_values || null,
                    });
                }
            }
            return out;
        },

        _renderBulkApply() {
            const fields = this.getSweepableFields();
            const options = fields.map(f => {
                const value = `${f.section}.${f.key}`;
                return `<option value="${this._escape(value)}" data-meta="${this._escape(JSON.stringify(f))}">${this._escape(f.group)} - ${this._escape(f.label)}</option>`;
            }).join('');

            const selectedCount = this.state.selectedModels.size;

            return `
                <div class="studio-section-title">Bulk Apply</div>
                <div class="field-card">
                    <div class="field-help">
                        Apply one parameter to <b>all</b>, <b>filtered</b>, or <b>selected</b> models.
                        This is the fastest way to set <code>jinja</code> / <code>parallel</code> across models.
                    </div>
                </div>

                <div class="field-card" style="margin-top: 1rem;">
                    <div class="field-grid">
                        <div class="form-group">
                            <label>Scope</label>
                            <select id="cfg-bulk-scope">
                                <option value="SELECTED">Selected (${selectedCount})</option>
                                <option value="FILTERED">Filtered (by text)</option>
                                <option value="ALL">All models</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Filter (for FILTERED)</label>
                            <input type="text" id="cfg-bulk-filter" placeholder="e.g. Qwen3">
                        </div>
                        <div class="form-group">
                            <label>Parameter</label>
                            <select id="cfg-bulk-param">
                                <option value="">Select...</option>
                                ${options}
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Action</label>
                            <label class="checkbox-label"><input type="checkbox" id="cfg-bulk-reset"> Reset (delete key)</label>
                        </div>
                        <div class="form-group">
                            <label>Value</label>
                            <div id="cfg-bulk-value-slot">
                                <input type="text" id="cfg-bulk-value" placeholder="Select a parameter">
                            </div>
                        </div>
                    </div>
                    <div class="form-actions" style="margin-top: 1rem;">
                        <button type="button" class="btn btn-primary" id="cfg-bulk-apply-btn">Apply</button>
                    </div>
                </div>
            `;
        },

        _bindBulkApplyUI() {
            const paramSel = document.getElementById('cfg-bulk-param');
            const valueSlot = document.getElementById('cfg-bulk-value-slot');
            const resetEl = document.getElementById('cfg-bulk-reset');

            const renderValueEditor = (meta) => {
                if (!valueSlot) return;
                if (!meta) {
                    valueSlot.innerHTML = `<input type="text" id="cfg-bulk-value" placeholder="Select a parameter">`;
                    return;
                }
                if (meta.ui === 'toggle' && meta.type === 'bool') {
                    valueSlot.innerHTML = `<label class="checkbox-label"><input type="checkbox" id="cfg-bulk-value"> Enabled</label>`;
                    return;
                }
                if (meta.ui === 'select' && Array.isArray(meta.enum_values)) {
                    valueSlot.innerHTML = `
                        <select id="cfg-bulk-value">
                            ${meta.enum_values.map(v => `<option value="${this._escape(v)}">${this._escape(v)}</option>`).join('')}
                        </select>
                    `;
                    return;
                }
                if (meta.ui === 'number') {
                    valueSlot.innerHTML = `<input type="number" id="cfg-bulk-value">`;
                    return;
                }
                valueSlot.innerHTML = `<input type="text" id="cfg-bulk-value">`;
            };

            if (paramSel) {
                paramSel.addEventListener('change', () => {
                    const opt = paramSel.selectedOptions?.[0];
                    if (!opt) return;
                    const metaRaw = opt.getAttribute('data-meta') || '';
                    let meta = null;
                    try { meta = JSON.parse(metaRaw); } catch {}
                    renderValueEditor(meta);
                });
            }

            const applyBtn = document.getElementById('cfg-bulk-apply-btn');
            if (applyBtn) {
                applyBtn.addEventListener('click', async () => {
                    const scope = document.getElementById('cfg-bulk-scope')?.value || 'SELECTED';
                    const filterString = document.getElementById('cfg-bulk-filter')?.value || '';
                    const param = document.getElementById('cfg-bulk-param')?.value || '';
                    const valEl = document.getElementById('cfg-bulk-value');
                    const doReset = !!resetEl?.checked;
                    if (!param) {
                        Toast.error('Select a parameter');
                        return;
                    }
                    const [section, key] = param.split('.', 2);
                    if (!section || !key) {
                        Toast.error('Invalid parameter');
                        return;
                    }

                    const target = (this.state.editTarget === 'override')
                        ? { kind: 'override', name: this.state.currentOverride }
                        : { kind: 'base' };
                    if (target.kind === 'override' && !target.name) {
                        Toast.error('Select an override (top-right) or switch to Base');
                        return;
                    }

                    let models = 'ALL';
                    let filter = null;
                    if (scope === 'SELECTED') {
                        const selected = Array.from(this.state.selectedModels);
                        if (selected.length === 0) {
                            Toast.error('No models selected');
                            return;
                        }
                        models = selected;
                    } else if (scope === 'FILTERED') {
                        models = 'ALL';
                        filter = filterString || '';
                        if (!filter) {
                            Toast.error('Enter a filter string');
                            return;
                        }
                    }

                    if (scope === 'ALL') {
                        if (!confirm('Apply this change to ALL models?')) return;
                    }

                    let value = null;
                    if (!doReset) {
                        if (valEl) {
                            if (valEl.type === 'checkbox') value = !!valEl.checked;
                            else value = valEl.value;
                        }
                        if (value === '' || value == null) {
                            Toast.error('Enter a value (or use Reset)');
                            return;
                        }

                        if (/^-?\\d+$/.test(String(value))) value = parseInt(String(value), 10);
                        else if (/^-?\\d+\\.\\d+$/.test(String(value))) value = parseFloat(String(value));
                    } else {
                        value = null;
                    }

                    try {
                        await API.post('/api/config/studio/bulk-apply', {
                            target,
                            models,
                            filter_string: filter,
                            section,
                            changes: { [key]: value },
                            context_override: this.state.currentOverride || null,
                        });
                        Toast.success('Applied');
                        await this.load(this.state.currentOverride);
                        this.state.view = 'bulk_apply';
                        this.renderNav();
                        this.render();
                    } catch (e) {
                        Toast.error(`Bulk apply failed: ${e.message}`);
                    }
                });
            }
        },

        getSelectedModels() {
            return Array.from(this.state.selectedModels);
        },
    };

    window.ConfigStudio = ConfigStudio;
})();
