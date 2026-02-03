/**
 * llama-suite Web UI - Main Application
 * 
 * Handles routing, API communication, WebSocket connection, and UI updates.
 */

// =============================================================================
// API Client
// =============================================================================

const API = {
    baseUrl: '',

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        if (config.body && typeof config.body === 'object') {
            config.body = JSON.stringify(config.body);
        }

        try {
            const response = await fetch(url, config);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || `HTTP ${response.status}`);
            }

            return data;
        } catch (error) {
            console.error(`API Error: ${endpoint}`, error);
            throw error;
        }
    },

    get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    },

    post(endpoint, body) {
        return this.request(endpoint, { method: 'POST', body });
    },

    put(endpoint, body) {
        return this.request(endpoint, { method: 'PUT', body });
    },

    delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }
};

// =============================================================================
// WebSocket Manager
// =============================================================================

class WebSocketManager {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 3000;
        this.handlers = new Map();
        this.connect();
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/progress`;

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.updateStatus(true);
                this.startPing();
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (e) {
                    // Ignore pong messages
                }
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateStatus(false);
                this.stopPing();
                setTimeout(() => this.connect(), this.reconnectInterval);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (e) {
            console.error('WebSocket connection failed:', e);
            setTimeout(() => this.connect(), this.reconnectInterval);
        }
    }

    startPing() {
        this.pingInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send('ping');
            }
        }, 30000);
    }

    stopPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
        }
    }

    updateStatus(connected) {
        const statusEl = document.getElementById('connection-status');
        if (statusEl) {
            const dot = statusEl.querySelector('.status-dot');
            const text = statusEl.querySelector('.status-text');
            if (connected) {
                dot.classList.add('connected');
                dot.classList.remove('disconnected');
                text.textContent = 'Connected';
            } else {
                dot.classList.remove('connected');
                dot.classList.add('disconnected');
                text.textContent = 'Disconnected';
            }
        }
    }

    handleMessage(data) {
        const handlers = this.handlers.get(data.type) || [];
        handlers.forEach(handler => handler(data));
    }

    on(type, handler) {
        if (!this.handlers.has(type)) {
            this.handlers.set(type, []);
        }
        this.handlers.get(type).push(handler);
    }

    off(type, handler) {
        if (this.handlers.has(type)) {
            const handlers = this.handlers.get(type).filter(h => h !== handler);
            this.handlers.set(type, handlers);
        }
    }
}

// =============================================================================
// Task Manager (Persistent Window)
// =============================================================================

class TaskManager {
    constructor() {
        this.window = document.getElementById('task-manager-window');
        this.list = document.getElementById('task-list');
        this.countBadge = document.getElementById('active-task-count');
        this.tasks = new Map(); // taskId -> element
        this.updateVisibility();
    }

    addTask(taskId, type, name) {
        if (this.tasks.has(taskId)) return;

        const item = document.createElement('div');
        item.className = 'task-item';
        item.id = `task-${taskId}`;

        let icon = 'fa-tasks';
        if (type === 'bench') icon = 'fa-tachometer-alt';
        else if (type === 'memory') icon = 'fa-memory';
        else if (type === 'eval') icon = 'fa-robot';
        else if (type === 'watcher') icon = 'fa-plug';
        else if (type === 'system') icon = 'fa-wrench';

        item.innerHTML = `
            <div class="task-icon"><i class="fas ${icon}"></i></div>
            <div class="task-details">
                <div class="task-name" title="${name}">${name}</div>
                <div class="task-status">Starting...</div>
                <div class="task-progress-track">
                    <div class="task-progress-fill" style="width: 0%"></div>
                </div>
            </div>
            <div class="task-actions">
                ${type !== 'watcher' ? `
                <button type="button" class="stop-btn" onclick="cancelTask('${taskId}')" title="Stop Task">
                    <i class="fas fa-stop"></i>
                </button>` : ''}
            </div>
        `;

        this.list.prepend(item);
        this.tasks.set(taskId, { element: item, type: type });
        this.updateVisibility();

        // Ensure window is visible immediately
        this.window.classList.remove('hidden');
    }

    updateTask(taskId, progress, statusMsg) {
        const task = this.tasks.get(taskId);
        if (!task) return;

        const el = task.element;
        const fill = el.querySelector('.task-progress-fill');
        const status = el.querySelector('.task-status');

        if (statusMsg) status.textContent = statusMsg;

        if (progress >= 0) {
            fill.className = 'task-progress-fill'; // remove indeterminate
            fill.style.width = `${progress}%`;
        } else {
            fill.className = 'task-progress-fill indeterminate';
        }
    }

    completeTask(taskId, success, message) {
        const task = this.tasks.get(taskId);
        if (!task) return;

        this.updateTask(taskId, 100, message || (success ? 'Completed' : 'Failed'));

        const el = task.element;
        if (success) {
            el.querySelector('.task-icon').innerHTML = '<i class="fas fa-check" style="color: var(--success)"></i>';
        } else {
            el.querySelector('.task-icon').innerHTML = '<i class="fas fa-times" style="color: var(--error)"></i>';
        }

        // Remove Stop button
        const actions = el.querySelector('.task-actions');
        if (actions) actions.innerHTML = '';

        // Auto-remove after delay
        setTimeout(() => {
            this.removeTask(taskId);
        }, 30000);
    }

    cancelTaskUI(taskId) {
        const task = this.tasks.get(taskId);
        if (!task) return;

        this.updateTask(taskId, 0, 'Cancelled');
        task.element.querySelector('.task-icon').innerHTML = '<i class="fas fa-ban" style="color: var(--text-muted)"></i>';

        // Remove from list after short delay
        setTimeout(() => this.removeTask(taskId), 5000);
    }

    removeTask(taskId) {
        const task = this.tasks.get(taskId);
        if (task) {
            task.element.remove();
            this.tasks.delete(taskId);
            this.updateVisibility();
        }
    }

    // Updates the persistent "Watcher" entry
    updateWatcherState(isRunning) {
        const watcherId = 'watcher-service';
        if (isRunning) {
            if (!this.tasks.has(watcherId)) {
                this.addTask(watcherId, 'watcher', 'LLM Endpoint (llama-swap)');
            }
            this.updateTask(watcherId, 100, 'Running');
            // Make progress bar solid green or pulsing
            const el = this.tasks.get(watcherId).element;
            const fill = el.querySelector('.task-progress-fill');
            fill.style.width = '100%';
            fill.style.background = 'var(--success)';
        } else {
            // Remove watcher task when stopped
            this.removeTask(watcherId);
        }
    }

    updateVisibility() {
        const count = this.tasks.size;
        // this.countBadge may be null if the element isn't found, check it
        if (this.countBadge) this.countBadge.textContent = count;

        if (count === 0) {
            this.window.classList.add('hidden');
        } else {
            this.window.classList.remove('hidden');
        }
    }
}

// =============================================================================
// Toast Notifications
// =============================================================================

const Toast = {
    container: null,

    init() {
        this.container = document.getElementById('toast-container');
    },

    show(message, type = 'info', duration = 4000) {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${this.getIcon(type)}</span>
            <span class="toast-message">${message}</span>
        `;

        this.container.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideIn 0.25s ease reverse';
            setTimeout(() => toast.remove(), 250);
        }, duration);
    },

    getIcon(type) {
        const icons = {
            success: '✓',
            error: '✕',
            warning: '⚠',
            info: 'ℹ'
        };
        return icons[type] || icons.info;
    },

    success(message) { this.show(message, 'success'); },
    error(message) { this.show(message, 'error'); },
    warning(message) { this.show(message, 'warning'); },
    info(message) { this.show(message, 'info'); }
};

// =============================================================================
// Modal
// =============================================================================

const Modal = {
    overlay: null,
    titleEl: null,
    bodyEl: null,
    footerEl: null,

    init() {
        this.overlay = document.getElementById('modal-overlay');
        this.titleEl = document.getElementById('modal-title');
        this.bodyEl = document.getElementById('modal-body');
        this.footerEl = document.getElementById('modal-footer');

        document.getElementById('modal-close').addEventListener('click', () => this.hide());
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) this.hide();
        });
    },

    show(title, bodyHtml, footerHtml = '') {
        this.titleEl.textContent = title;
        this.bodyEl.innerHTML = bodyHtml;
        this.footerEl.innerHTML = footerHtml;
        this.overlay.classList.add('active');
    },

    hide() {
        this.overlay.classList.remove('active');
    }
};

// =============================================================================
// App State
// =============================================================================

const App = {
    currentSection: 'dashboard',
    currentOverride: '',
    ws: null,
    baseConfigOriginal: null,
    missingModelsForDownload: [],
    outputContainers: {},
    currentBenchTaskId: null,
    currentMemoryTaskId: null,
    currentEvalHarnessTaskId: null,
    currentEvalCustomTaskId: null,
    currentWatcherTaskId: null,
    resultsState: {
        bench: { data: [], sort: { col: 'Timestamp', asc: false }, filter: { search: '', status: '' }, selected: new Set() },
        memory: { data: [], sort: { col: 'Timestamp', asc: false }, filter: { search: '', status: '' }, selected: new Set() },
        eval: { data: [], sort: { col: 'gen_judge_score', asc: false }, filter: { search: '' }, selected: new Set() }
    },

    async init() {
        Toast.init();
        Modal.init();
        this.ws = new WebSocketManager();

        this.setupNavigation();
        this.setupTabs();
        this.setupDirtyTracking();
        this.setupWebSocketHandlers();
        await this.loadInitialData();

        // Handle initial route
        this.handleRoute();
        window.addEventListener('hashchange', () => this.handleRoute());
    },

    setupDirtyTracking() {
        const editor = document.getElementById('config-editor');
        if (!editor) return;
        if (this._dirtyTrackingBound) return;
        this._dirtyTrackingBound = true;

        editor.addEventListener('input', () => this.updateBaseConfigSaveButton());
        this.updateBaseConfigSaveButton();
    },

    updateBaseConfigSaveButton() {
        const btn = document.getElementById('btn-save-config');
        const editor = document.getElementById('config-editor');
        if (!btn || !editor) return;

        const original = this.baseConfigOriginal ?? '';
        const dirty = editor.value !== original;
        btn.disabled = !dirty;
        btn.textContent = dirty ? 'Save Changes' : 'Saved';
    },

    setupNavigation() {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const section = item.dataset.section;
                window.location.hash = section;
            });
        });

        // Override selector
        document.getElementById('current-override').addEventListener('change', async (e) => {
            this.currentOverride = e.target.value;
            try {
                localStorage.setItem('llama-suite.override', this.currentOverride || '');
            } catch {
                // Ignore storage errors (private mode / restricted contexts)
            }
            this.updateOverrideActionsUI();
            await this.loadDownloadModelsUI();
            await this.refreshCurrentSection();
        });
    },

    setupTabs() {
        document.querySelectorAll('.section-tabs').forEach(tabContainer => {
            tabContainer.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const tab = btn.dataset.tab;

                    // Update active button
                    tabContainer.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');

                    // Show corresponding content
                    const section = tabContainer.closest('.section');
                    section.querySelectorAll('.tab-content').forEach(content => {
                        content.classList.toggle('active', content.id === tab);
                    });
                });
            });
        });
    },

    setupWebSocketHandlers() {
        this.currentEvalCustomTaskId = null;
        this.currentWatcherTaskId = 'watcher-service'; // Fixed ID for watcher

        // Initialize Task Manager
        this.taskManager = new TaskManager();

        // Initialize WS handlers
        this.ws.on('log', (data) => {
            const line = data.message ?? data.line ?? '';
            if (line) {
                this.appendOutput(data.task_id, line, data.level || 'info');
            }
        });

        this.ws.on('progress', (data) => {
            // Update persistent task
            const progress = data.percentage ?? data.progress ?? 0;
            const message = data.message ?? '';
            this.taskManager.updateTask(data.task_id, progress, message);
        });

        this.ws.on('complete', (data) => {
            Toast.success('Task completed');
            this.refreshDashboard();

            this.taskManager.completeTask(data.task_id, data.success, data.success ? 'Completed' : 'Failed');
            this.checkAndResetTaskState(data.task_id);
        });

        this.ws.on('cancelled', (data) => {
            Toast.info('Task cancelled');
            this.refreshDashboard();

            this.taskManager.cancelTaskUI(data.task_id);

            this.checkAndResetTaskState(data.task_id);
        });
    },

    updateEndpointStatusUI(isRunning, statusTextOverride = null) {
        const statusText = statusTextOverride || (isRunning ? 'Running' : 'Stopped');

        const statusEl = document.getElementById('watcher-status');
        if (statusEl) {
            statusEl.textContent = statusText;
            statusEl.className = `status-badge ${isRunning ? 'status-running' : 'status-stopped'}`;
            if (statusTextOverride === 'Error') {
                statusEl.className = 'status-badge status-error';
            }
        }

        const startBtn = document.getElementById('btn-watcher-start');
        const stopBtn = document.getElementById('btn-watcher-stop');
        if (startBtn) startBtn.disabled = isRunning;
        if (stopBtn) stopBtn.disabled = !isRunning;

        const qsStatusEl = document.getElementById('dashboard-endpoint-status');
        if (qsStatusEl) {
            qsStatusEl.textContent = statusText;
            qsStatusEl.className = `status-badge ${isRunning ? 'status-running' : 'status-stopped'}`;
            if (statusTextOverride === 'Error') {
                qsStatusEl.className = 'status-badge status-error';
            }
        }

        const qsStart = document.getElementById('btn-quickstart-endpoint-start');
        const qsStop = document.getElementById('btn-quickstart-endpoint-stop');
        if (qsStart) qsStart.disabled = isRunning;
        if (qsStop) qsStop.disabled = !isRunning;

        if (this.taskManager) {
            this.taskManager.updateWatcherState(isRunning);
        }
    },

    updateOpenWebUIStatusUI(isRunning, statusTextOverride = null) {
        const statusText = statusTextOverride || (isRunning ? 'Running' : 'Stopped');
        const cls = statusTextOverride === 'Error'
            ? 'status-badge status-error'
            : `status-badge ${isRunning ? 'status-running' : 'status-stopped'}`;

        const statusEl = document.getElementById('dashboard-openwebui-status');
        if (statusEl) {
            statusEl.textContent = statusText;
            statusEl.className = cls;
        }

        const startBtn = document.getElementById('btn-quickstart-openwebui-start');
        const stopBtn = document.getElementById('btn-quickstart-openwebui-stop');
        if (startBtn) startBtn.disabled = isRunning;
        if (stopBtn) stopBtn.disabled = !isRunning;

        const stopBtnSystem = document.getElementById('btn-openwebui-stop');
        if (stopBtnSystem) stopBtnSystem.disabled = !isRunning;
    },

    findOutputContainer(taskId) {
        if (taskId === this.currentBenchTaskId) return 'bench-output';
        if (taskId === this.currentMemoryTaskId) return 'memory-output';
        if (taskId === this.currentEvalHarnessTaskId) return 'eval-output';
        if (taskId === this.currentEvalCustomTaskId) return 'eval-output';
        if (taskId === this.currentWatcherTaskId) return 'watcher-output';
        return null;
    },

    checkAndResetTaskState(taskId) {
        if (taskId === this.currentBenchTaskId) {
            this.currentBenchTaskId = null;
        } else if (taskId === this.currentMemoryTaskId) {
            this.currentMemoryTaskId = null;
        } else if (taskId === this.currentEvalHarnessTaskId) {
            this.currentEvalHarnessTaskId = null;
        } else if (taskId === this.currentEvalCustomTaskId) {
            this.currentEvalCustomTaskId = null;
        } else if (taskId === this.currentWatcherTaskId) {
            this.currentWatcherTaskId = null;
            this.updateEndpointStatusUI(false);
        }
    },

    setTaskRunningState(type, isRunning) {
        const startBtn = document.getElementById(`btn-${type}-start`);
        const stopBtn = document.getElementById(`btn-${type}-stop`);
        if (startBtn && stopBtn) {
            startBtn.style.display = isRunning ? 'none' : 'inline-block';
            stopBtn.style.display = isRunning ? 'inline-block' : 'none';
        }
    },

    setWatcherRunningState(isRunning) {
        this.updateEndpointStatusUI(isRunning);
    },

    appendOutput(taskId, line, level = 'info') {
        // Try to find output container by task ID mapping first
        let container = this.taskContainers?.get(taskId);

        // Fall back to well-known task output containers (even if section isn't active)
        if (!container) {
            const outputId = this.findOutputContainer(taskId);
            if (outputId) {
                container = document.getElementById(outputId);
            }
        }

        // Fall back to active output container
        if (!container) {
            container = this.getActiveOutputContainer();
        }

        if (container) {
            const lineEl = document.createElement('div');
            lineEl.className = `log-line ${level}`;
            lineEl.textContent = line;
            container.appendChild(lineEl);
            container.scrollTop = container.scrollHeight;
        }
    },

    registerTaskContainer(taskId, containerId) {
        if (!this.taskContainers) {
            this.taskContainers = new Map();
        }
        const container = document.getElementById(containerId);
        if (container) {
            this.taskContainers.set(taskId, container);
        }
    },

    unregisterTaskContainer(taskId) {
        if (this.taskContainers) {
            this.taskContainers.delete(taskId);
        }
    },

    getActiveOutputContainer() {
        const outputIds = ['bench-output', 'memory-output', 'eval-output', 'watcher-output', 'system-output'];
        for (const id of outputIds) {
            const el = document.getElementById(id);
            if (el && el.closest('.section.active')) {
                return el;
            }
        }
        return null;
    },

    handleRoute() {
        const hash = window.location.hash.slice(1) || 'dashboard';
        this.navigateTo(hash);
    },

    navigateTo(section) {
        this.currentSection = section;

        // Update nav
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.toggle('active', item.dataset.section === section);
        });

        // Update sections
        document.querySelectorAll('.section').forEach(s => {
            s.classList.toggle('active', s.id === `section-${section}`);
        });

        // Update title
        const titles = {
            dashboard: 'Dashboard',
            config: 'Configuration',
            models: 'Models',
            bench: 'Benchmark',
            memory: 'Memory Scan',
            eval: 'Evaluation',
            watcher: 'LLM Endpoint',
            results: 'Results',
            system: 'System'
        };
        document.getElementById('page-title').textContent = titles[section] || section;

        // Load section data
        this.loadSectionData(section);
    },

    async loadInitialData() {
        await this.loadOverrides();
        await this.loadSystemInfo();
    },

    async loadOverrides() {
        try {
            const data = await API.get('/api/config/overrides');
            const select = document.getElementById('current-override');
            select.innerHTML = '<option value="">None (base only)</option>';
            data.overrides.forEach(o => {
                const option = document.createElement('option');
                option.value = o.name;
                option.textContent = o.name;
                select.appendChild(option);
            });

            // Restore last selection (best-effort)
            let desired = this.currentOverride || '';
            if (!desired) {
                try {
                    desired = localStorage.getItem('llama-suite.override') || '';
                } catch {
                    desired = '';
                }
            }
            const hasDesired = desired && Array.from(select.options).some(opt => opt.value === desired);
            select.value = hasDesired ? desired : '';
            this.currentOverride = select.value;
            this.updateOverrideActionsUI();
        } catch (e) {
            console.error('Failed to load overrides:', e);
        }
    },

    updateOverrideActionsUI() {
        const hasOverride = !!this.currentOverride;
        const editBtn = document.getElementById('btn-override-edit');
        const dupBtn = document.getElementById('btn-override-duplicate');
        if (editBtn) editBtn.disabled = !hasOverride;
        if (dupBtn) dupBtn.disabled = !hasOverride;
    },

    async loadSystemInfo() {
        try {
            const info = await API.get('/api/system/info');
            document.getElementById('info-platform').textContent = info.platform;
            document.getElementById('info-python').textContent = info.python_version;
            document.getElementById('info-root').textContent = info.project_root;
            document.getElementById('info-venv').textContent = info.venv_exists ? 'Active' : 'Not found';
        } catch (e) {
            console.error('Failed to load system info:', e);
        }
    },

    async loadSectionData(section) {
        switch (section) {
            case 'dashboard':
                await this.refreshDashboard();
                break;
            case 'config':
                await this.loadConfig();
                break;
            case 'models':
                await this.loadModels();
                break;
            case 'eval':
                await this.loadDatasets();
                break;
            case 'watcher':
                await this.loadWatcherStatus();
                break;
            case 'results':
                await this.loadResults();
                break;
            case 'system':
                await this.loadSystemSection();
                break;
        }

        // Load model list for dropdowns
        if (['bench', 'memory', 'eval'].includes(section)) {
            await this.loadModelDropdowns();
        }
    },

    async refreshCurrentSection() {
        await this.loadSectionData(this.currentSection);
    },

    async loadSystemSection() {
        await this.loadOpenWebUIStatus();
        await this.loadDownloadModelsUI();
    },

    goToDownloadModels() {
        Modal.hide();
        window.location.hash = 'system';
        setTimeout(() => {
            const form = document.getElementById('download-form');
            if (form) form.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 50);
    },

    async loadDownloadModelsUI() {
        const listEl = document.getElementById('download-models-list');
        const helpEl = document.getElementById('download-models-help');
        const btn = document.getElementById('btn-download-models');
        if (!listEl || !helpEl || !btn) return;

        listEl.innerHTML = '';
        helpEl.textContent = 'Loading missing models…';
        btn.disabled = true;

        try {
            const url = this.currentOverride
                ? `/api/models?override=${encodeURIComponent(this.currentOverride)}`
                : '/api/models';
            const data = await API.get(url);
            const models = data.models || [];

            this.missingModelsForDownload = models
                .filter(m => !m.disabled && !m.model_exists)
                .map(m => m.name);

            if (this.missingModelsForDownload.length === 0) {
                const msg = document.createElement('div');
                msg.className = 'empty-message';
                msg.style.padding = '0.75rem';
                msg.textContent = 'No missing models for this override.';
                listEl.appendChild(msg);
                helpEl.textContent = 'Tip: use “Force re-download” to fetch again even if files already exist.';
                btn.textContent = '⬇ Download models';
                btn.disabled = !document.getElementById('download-force')?.checked;
                return;
            }

            for (const name of this.missingModelsForDownload) {
                const label = document.createElement('label');
                label.className = 'checklist-item';

                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.className = 'download-model-checkbox';
                cb.value = name;
                cb.checked = true;

                const text = document.createElement('span');
                text.textContent = name;

                label.appendChild(cb);
                label.appendChild(text);
                listEl.appendChild(label);
            }

            const updateHelp = () => {
                const selected = Array.from(document.querySelectorAll('.download-model-checkbox:checked')).map(el => el.value);
                helpEl.textContent = `${selected.length}/${this.missingModelsForDownload.length} selected`;
                btn.disabled = selected.length === 0;
                btn.textContent = selected.length === 0 ? '⬇ Download missing models' : `⬇ Download ${selected.length} missing model(s)`;
            };

            listEl.onchange = updateHelp;
            updateHelp();
        } catch (e) {
            helpEl.textContent = `Failed to load missing models: ${e.message}`;
            btn.textContent = '⬇ Download models';
            btn.disabled = false;
        }
    },

    async refreshDashboard() {
        try {
            // Load stats
            const models = await API.get('/api/models');
            document.getElementById('stat-models').textContent = models.count;

            const files = await API.get('/api/models/files/available');
            document.getElementById('stat-files').textContent = files.files.length;

            const results = await API.get('/api/results');
            const benchRuns = results.types.find(t => t.type === 'bench');
            const evalRuns = results.types.find(t => t.type === 'eval');
            document.getElementById('stat-bench-runs').textContent = benchRuns?.run_count || 0;
            document.getElementById('stat-eval-runs').textContent = evalRuns?.run_count || 0;

            // Load active tasks
            const tasks = await API.get('/api/system/tasks');
            const runningTasks = tasks.tasks.filter(t => t.status === 'running');
            const taskList = document.getElementById('active-tasks');

            if (runningTasks.length === 0) {
                taskList.innerHTML = '<p class="empty-message">No active tasks</p>';
            } else {
                const typeLabel = (t) => (t.task_type === 'watcher' ? 'endpoint' : t.task_type);
                taskList.innerHTML = runningTasks.map(t => `
                    <div class="task-item">
                        <div class="task-spinner"></div>
                        <div class="task-info">
                            <div class="task-name">${t.description}</div>
                            <div class="task-progress">${typeLabel(t)} • Started ${new Date(t.started_at).toLocaleTimeString()}</div>
                        </div>
                        <button class="btn btn-secondary" onclick="App.cancelTask('${t.task_id}')">Cancel</button>
                    </div>
                `).join('');
            }

            // Update endpoint status for the Quick Start card (and the Endpoint section badge).
            try {
                const endpointStatus = await API.get('/api/watcher/status');
                this.updateEndpointStatusUI(!!endpointStatus.running);
            } catch (e2) {
                this.updateEndpointStatusUI(false, 'Error');
            }

            await this.loadOpenWebUIStatus();
        } catch (e) {
            console.error('Failed to refresh dashboard:', e);
        }
    },

    async loadConfig() {
        try {
            const data = await API.get('/api/config');
            document.getElementById('config-editor').value = data.content;
            this.baseConfigOriginal = data.content;
            this.updateBaseConfigSaveButton();

            // Load overrides list
            const overrides = await API.get('/api/config/overrides');
            const list = document.getElementById('overrides-list');
            if (overrides.overrides.length === 0) {
                list.innerHTML = '<p class="empty-message">No override files found</p>';
            } else {
                list.innerHTML = overrides.overrides.map(o => `
                    <div class="list-item" onclick="App.editOverride('${o.name}')">
                        <span>${o.filename}</span>
                        <button class="btn btn-secondary" onclick="event.stopPropagation(); App.deleteOverride('${o.name}')">Delete</button>
                    </div>
                `).join('');
            }

            // Load effective config
            await this.loadEffectiveConfig();
        } catch (e) {
            console.error('Failed to load config:', e);
            Toast.error('Failed to load configuration');
        }
    },

    async loadEffectiveConfig() {
        try {
            const override = this.currentOverride || undefined;
            const url = override ? `/api/config/effective?override=${override}` : '/api/config/effective';
            const data = await API.get(url);
            document.getElementById('effective-config').textContent = data.yaml || JSON.stringify(data.config, null, 2);
        } catch (e) {
            console.error('Failed to load effective config:', e);
        }
    },

    async loadModels() {
        try {
            const url = this.currentOverride
                ? `/api/models?override=${this.currentOverride}`
                : '/api/models';
            const data = await API.get(url);

            // Store models for later use
            this.modelsData = data.models;

            const grid = document.getElementById('models-grid');
            const searchInput = document.getElementById('model-search');

            const renderModels = (filter = '') => {
                const filtered = data.models.filter(m =>
                    m.name.toLowerCase().includes(filter.toLowerCase())
                );

                grid.innerHTML = filtered.map(m => `
                    <div class="model-card ${m.disabled ? 'disabled' : ''}" onclick="App.editModel('${m.name}')">
                        <div class="model-card-header">
                            <span class="model-name">${m.name}</span>
                            <span class="model-status ${m.model_exists ? '' : 'missing'}">
                                ${m.model_exists ? 'Ready' : 'Missing'}
                            </span>
                        </div>
                        <div class="model-details">
                            <div class="model-detail">
                                <span class="model-detail-label">CTX:</span>
                                <span>${m.ctx_size || '—'}</span>
                            </div>
                            <div class="model-detail">
                                <span class="model-detail-label">GPU:</span>
                                <span>${m.gpu_layers ?? 'auto'}</span>
                            </div>
                            <div class="model-detail">
                                <span class="model-detail-label">Threads:</span>
                                <span>${m.threads || 'auto'}</span>
                            </div>
                            ${m.has_draft_model ? '<div class="model-detail">📝 Draft</div>' : ''}
                            ${m.disabled ? '<div class="model-detail">⏸ Disabled</div>' : ''}
                        </div>
                        <div class="model-path" style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.5rem; word-break: break-all;">
                            ${m.model_path ? m.model_path.split(/[/\\]/).pop() : 'No path'}
                        </div>
                    </div>
                `).join('');
            };

            searchInput.addEventListener('input', (e) => renderModels(e.target.value));
            renderModels();
        } catch (e) {
            console.error('Failed to load models:', e);
        }
    },

    async editModel(name) {
        try {
            const url = this.currentOverride
                ? `/api/models/${encodeURIComponent(name)}?override=${this.currentOverride}`
                : `/api/models/${encodeURIComponent(name)}`;
            const data = await API.get(url);
            const config = data.config;
            const cmd = data.cmd || {};
            const sampling = config.sampling || {};

            const otherModels = (this.modelsData || []).filter(m => m.name !== name);

            Modal.show(`Edit Model: ${name}`, `
                <div class="model-editor">
                    <div class="form-grid" style="grid-template-columns: 1fr 1fr;">
                        <div class="form-group">
                            <label>Context Size (ctx-size)</label>
                            <input type="number" id="edit-ctx-size" value="${cmd['ctx-size'] || ''}" placeholder="e.g., 8192">
                        </div>
                        <div class="form-group">
                            <label>GPU Layers (gpu-layers)</label>
                            <input type="number" id="edit-gpu-layers" value="${cmd['gpu-layers'] ?? ''}" placeholder="-1 = all">
                        </div>
                        <div class="form-group">
                            <label>Threads</label>
                            <input type="number" id="edit-threads" value="${cmd['threads'] || ''}" placeholder="auto">
                        </div>
                        <div class="form-group">
                            <label>HF Tokenizer</label>
                            <input type="text" id="edit-tokenizer" value="${config.hf_tokenizer_for_model || ''}" placeholder="e.g., Qwen/Qwen3-8B">
                        </div>
                    </div>
                    
                    <h4 style="margin: 1rem 0 0.5rem;">Sampling Parameters</h4>
                    <div class="form-grid" style="grid-template-columns: repeat(3, 1fr);">
                        <div class="form-group">
                            <label>Temperature</label>
                            <input type="number" step="0.1" id="edit-temp" value="${sampling['temp'] ?? sampling['temperature'] ?? ''}" placeholder="0.7">
                        </div>
                        <div class="form-group">
                            <label>Top-P</label>
                            <input type="number" step="0.01" id="edit-top-p" value="${sampling['top-p'] ?? ''}" placeholder="0.9">
                        </div>
                        <div class="form-group">
                            <label>Top-K</label>
                            <input type="number" id="edit-top-k" value="${sampling['top-k'] ?? ''}" placeholder="40">
                        </div>
                        <div class="form-group">
                            <label>Min-P</label>
                            <input type="number" step="0.01" id="edit-min-p" value="${sampling['min-p'] ?? ''}" placeholder="0.05">
                        </div>
                        <div class="form-group">
                            <label>Repeat Penalty</label>
                            <input type="number" step="0.1" id="edit-repeat-penalty" value="${sampling['repeat-penalty'] ?? ''}" placeholder="1.1">
                        </div>
                    </div>
                    
                    <div class="form-group" style="margin-top: 1rem;">
                        <label class="checkbox-label">
                            <input type="checkbox" id="edit-disabled" ${config.disabled ? 'checked' : ''}> Disabled
                        </label>
                    </div>
                    
                    <details style="margin-top: 1rem;">
                        <summary style="cursor: pointer; color: var(--text-secondary);">Full Configuration (JSON)</summary>
                        <pre class="code-display" style="max-height: 200px; margin-top: 0.5rem;">${JSON.stringify(config, null, 2)}</pre>
                    </details>
                </div>
            `, `
                <button class="btn btn-danger" onclick="App.deleteModel('${name}')" style="margin-right: auto;">🗑 Delete</button>
                <button class="btn btn-secondary" onclick="App.showCopyDialog('${name}')">📋 Copy To...</button>
                <button class="btn btn-secondary" onclick="Modal.hide()">Cancel</button>
                <button class="btn btn-primary" onclick="App.saveModel('${name}')">Save Changes</button>
            `);
        } catch (e) {
            Toast.error(`Failed to load model: ${e.message}`);
        }
    },

    async saveModel(name) {
        try {
            const cmd = {};
            const sampling = {};

            const ctxSize = document.getElementById('edit-ctx-size').value;
            const gpuLayers = document.getElementById('edit-gpu-layers').value;
            const threads = document.getElementById('edit-threads').value;
            const tokenizer = document.getElementById('edit-tokenizer').value;
            const temp = document.getElementById('edit-temp').value;
            const topP = document.getElementById('edit-top-p').value;
            const topK = document.getElementById('edit-top-k').value;
            const minP = document.getElementById('edit-min-p').value;
            const repeatPenalty = document.getElementById('edit-repeat-penalty').value;
            const disabled = document.getElementById('edit-disabled').checked;

            if (ctxSize) cmd['ctx-size'] = parseInt(ctxSize);
            if (gpuLayers !== '') cmd['gpu-layers'] = parseInt(gpuLayers);
            if (threads) cmd['threads'] = parseInt(threads);

            if (temp) sampling['temp'] = parseFloat(temp);
            if (topP) sampling['top-p'] = parseFloat(topP);
            if (topK) sampling['top-k'] = parseInt(topK);
            if (minP) sampling['min-p'] = parseFloat(minP);
            if (repeatPenalty) sampling['repeat-penalty'] = parseFloat(repeatPenalty);

            await API.put(`/api/models/${encodeURIComponent(name)}`, {
                cmd: Object.keys(cmd).length > 0 ? cmd : undefined,
                sampling: Object.keys(sampling).length > 0 ? sampling : undefined,
                hf_tokenizer_for_model: tokenizer || undefined,
                disabled: disabled
            });

            Toast.success(`Model "${name}" updated`);
            Modal.hide();
            await this.loadModels();
        } catch (e) {
            Toast.error(`Failed to save: ${e.message}`);
        }
    },

    async showCopyDialog(sourceName) {
        Modal.hide();

        // Load all models except the source
        const otherModels = (this.modelsData || []).filter(m => m.name !== sourceName);

        Modal.show(`Copy Settings from: ${sourceName}`, `
            <div class="form-grid" style="grid-template-columns: 1fr;">
                <div class="form-group">
                    <label>Select target models:</label>
                    <select id="copy-targets" multiple style="height: 200px; width: 100%;">
                        ${otherModels.map(m => `<option value="${m.name}">${m.name}</option>`).join('')}
                    </select>
                    <small style="color: var(--text-muted);">Hold Ctrl/Cmd to select multiple models</small>
                </div>
                
                <h4 style="margin: 1rem 0 0.5rem;">What to copy:</h4>
                <div class="form-group">
                    <label class="checkbox-label">
                        <input type="checkbox" id="copy-cmd" checked> 
                        <strong>Command params</strong> (ctx-size, gpu-layers, threads, batch-size)
                    </label>
                </div>
                <div class="form-group">
                    <label class="checkbox-label">
                        <input type="checkbox" id="copy-sampling" checked> 
                        <strong>Sampling params</strong> (temp, top-p, top-k, min-p, repeat-penalty)
                    </label>
                </div>
            </div>
        `, `
            <button class="btn btn-secondary" onclick="App.editModel('${sourceName}')">← Back</button>
            <button class="btn btn-primary" onclick="App.copyModelParams('${sourceName}')">Copy to Selected</button>
        `);
    },

    async copyModelParams(sourceName) {
        try {
            const targetSelect = document.getElementById('copy-targets');
            const targets = Array.from(targetSelect.selectedOptions).map(o => o.value);

            if (targets.length === 0) {
                Toast.warning('Please select at least one target model');
                return;
            }

            const copyCmd = document.getElementById('copy-cmd').checked;
            const copySampling = document.getElementById('copy-sampling').checked;

            await API.post(`/api/models/${encodeURIComponent(sourceName)}/copy-to`, {
                target_models: targets,
                copy_cmd: copyCmd,
                copy_sampling: copySampling
            });

            Toast.success(`Copied params from "${sourceName}" to ${targets.length} model(s)`);
            Modal.hide();
            await this.loadModels();
        } catch (e) {
            Toast.error(`Failed to copy: ${e.message}`);
        }
    },

    async deleteModel(name) {
        Modal.hide();

        // Get the model info to show file path
        const model = (this.modelsData || []).find(m => m.name === name);
        const modelPath = model?.model_path || 'Unknown';
        const fileName = modelPath.split(/[/\\]/).pop();

        Modal.show(`Delete Model: ${name}`, `
            <div class="form-grid" style="grid-template-columns: 1fr;">
                <p style="margin-bottom: 1rem;">
                    <strong>Model file:</strong> <code>${fileName}</code>
                </p>
                
                <div class="form-group">
                    <label class="checkbox-label" style="padding: 1rem; border: 1px solid var(--border-light); border-radius: var(--radius-md);">
                        <input type="checkbox" id="delete-gguf-file"> 
                        <span><strong>Also delete GGUF file from disk</strong><br>
                        <small style="color: var(--text-muted);">⚠️ This is permanent and cannot be undone!</small></span>
                    </label>
                </div>
                
                <div id="gguf-warning" style="display: none; padding: 1rem; background: rgba(255, 100, 100, 0.1); border: 1px solid rgba(255, 100, 100, 0.3); border-radius: var(--radius-md); margin-top: 0.5rem;">
                    <strong>⚠️ Warning:</strong> Other models also use this GGUF file:
                    <ul id="gguf-dependent-models" style="margin: 0.5rem 0 0 1.5rem; padding: 0;"></ul>
                </div>
            </div>
        `, `
            <button class="btn btn-secondary" onclick="Modal.hide()">Cancel</button>
            <button class="btn btn-danger" onclick="App.confirmDeleteModel('${name}')">Delete Model</button>
        `);

        // Add event listener to check dependencies when checkbox is toggled
        const checkbox = document.getElementById('delete-gguf-file');
        checkbox.addEventListener('change', async () => {
            const warningDiv = document.getElementById('gguf-warning');
            if (checkbox.checked && model?.model_path) {
                // Check for other models using this file
                const otherModels = (this.modelsData || []).filter(m =>
                    m.name !== name && m.model_path === model.model_path
                );

                if (otherModels.length > 0) {
                    const list = document.getElementById('gguf-dependent-models');
                    list.innerHTML = otherModels.map(m => `<li>${m.name}</li>`).join('');
                    warningDiv.style.display = 'block';
                } else {
                    warningDiv.style.display = 'none';
                }
            } else {
                warningDiv.style.display = 'none';
            }
        });
    },

    async confirmDeleteModel(name) {
        const deleteFile = document.getElementById('delete-gguf-file').checked;

        // Final confirmation for file deletion
        if (deleteFile) {
            if (!confirm('⚠️ FINAL WARNING: This will PERMANENTLY DELETE the GGUF file!\\n\\nAre you absolutely sure?')) {
                return;
            }
        }

        try {
            const url = deleteFile
                ? `/api/models/${encodeURIComponent(name)}?delete_file=true`
                : `/api/models/${encodeURIComponent(name)}`;

            const result = await API.delete(url);

            if (result.deleted_file) {
                Toast.success(`Model "${name}" deleted (file also removed)`);
            } else {
                Toast.success(`Model "${name}" removed from config`);
            }

            // Warn about other models that used the same file
            if (result.other_models_using_file && result.other_models_using_file.length > 0 && result.deleted_file) {
                Toast.warning(`Note: ${result.other_models_using_file.length} other config(s) still reference this file`);
            }

            Modal.hide();
            await this.loadModels();
            await this.refreshDashboard();
        } catch (e) {
            Toast.error(`Failed to delete: ${e.message}`);
        }
    },

    async loadModelDropdowns() {
        try {
            const url = this.currentOverride
                ? `/api/models?override=${this.currentOverride}`
                : '/api/models';
            const data = await API.get(url);

            const selects = [
                'bench-model', 'memory-model', 'eval-model', 'custom-model'
            ];

            selects.forEach(id => {
                const select = document.getElementById(id);
                if (select) {
                    const currentValue = select.value;
                    select.innerHTML = '<option value="">All models</option>';
                    data.models.forEach(m => {
                        if (!m.disabled) {
                            const option = document.createElement('option');
                            option.value = m.name;
                            option.textContent = m.name;
                            select.appendChild(option);
                        }
                    });
                    select.value = currentValue;
                }
            });
        } catch (e) {
            console.error('Failed to load model dropdowns:', e);
        }
    },

    async loadDatasets() {
        try {
            const data = await API.get('/api/eval/datasets');
            const select = document.getElementById('custom-dataset');
            select.innerHTML = '';
            data.datasets.forEach(d => {
                const option = document.createElement('option');
                option.value = d.name;
                option.textContent = `${d.name} (${d.task_count || '?'} tasks)`;
                select.appendChild(option);
            });
        } catch (e) {
            console.error('Failed to load datasets:', e);
        }
    },

    async loadWatcherStatus() {
        try {
            const data = await API.get('/api/watcher/status');
            this.updateEndpointStatusUI(!!data.running);
        } catch (e) {
            console.error('Failed to get watcher status:', e);
            this.updateEndpointStatusUI(false, 'Error');
        }
    },

    async loadOpenWebUIStatus() {
        try {
            const data = await API.get('/api/system/openwebui/status');
            if (!data.runtime_found) {
                this.updateOpenWebUIStatusUI(false, 'Unavailable');
                return;
            }
            this.updateOpenWebUIStatusUI(!!data.running);
        } catch (e) {
            console.error('Failed to get Open WebUI status:', e);
            this.updateOpenWebUIStatusUI(false, 'Error');
        }
    },

    // Backwards-compat helper used by older code paths
    setWatcherUIState(isRunning) {
        this.updateEndpointStatusUI(isRunning);
    },

    async loadResults() {
        try {
            // Setup handlers once
            if (!this._resultsHandlersSetup) {
                this.setupResultsHandlers();
                this._resultsHandlersSetup = true;
            }

            const [benchRes, memoryRes, evalRes] = await Promise.all([
                API.get('/api/results/bench/merged'),
                API.get('/api/results/memory/merged'),
                API.get('/api/results/eval/merged')
            ]);

            this.resultsState.bench.data = benchRes.results || [];
            this.resultsState.memory.data = memoryRes.results || [];
            this.resultsState.eval.data = evalRes.results || [];

            this.renderBenchResults();
            this.renderMemoryResults();
            this.renderEvalResults();
        } catch (e) {
            console.error('Failed to load results:', e);
            Toast.error('Failed to load results');
        }
    },

    setupResultsHandlers() {
        // Bench handlers
        document.getElementById('bench-search').addEventListener('input', (e) => {
            this.resultsState.bench.filter.search = e.target.value;
            this.renderBenchResults();
        });
        document.getElementById('bench-status-filter').addEventListener('change', (e) => {
            this.resultsState.bench.filter.status = e.target.value;
            this.renderBenchResults();
        });
        document.getElementById('bench-select-all').addEventListener('change', (e) => {
            this.toggleSelectAll('bench', e.target.checked);
        });
        document.querySelectorAll('#bench-results-table th.sortable').forEach(th => {
            th.addEventListener('click', () => this.handleResultsSort('bench', th.dataset.sort));
        });
        document.getElementById('btn-bench-compare').addEventListener('click', () => this.compareResults('bench'));

        // Memory handlers
        document.getElementById('memory-search').addEventListener('input', (e) => {
            this.resultsState.memory.filter.search = e.target.value;
            this.renderMemoryResults();
        });
        document.getElementById('memory-status-filter').addEventListener('change', (e) => {
            this.resultsState.memory.filter.status = e.target.value;
            this.renderMemoryResults();
        });
        document.getElementById('memory-select-all').addEventListener('change', (e) => {
            this.toggleSelectAll('memory', e.target.checked);
        });
        document.querySelectorAll('#memory-results-table th.sortable').forEach(th => {
            th.addEventListener('click', () => this.handleResultsSort('memory', th.dataset.sort));
        });
        document.getElementById('btn-memory-compare').addEventListener('click', () => this.compareResults('memory'));

        // Eval handlers
        document.getElementById('eval-search').addEventListener('input', (e) => {
            this.resultsState.eval.filter.search = e.target.value;
            this.renderEvalResults();
        });
        document.getElementById('eval-select-all').addEventListener('change', (e) => {
            this.toggleSelectAll('eval', e.target.checked);
        });
        document.querySelectorAll('#eval-results-table th.sortable').forEach(th => {
            th.addEventListener('click', () => this.handleResultsSort('eval', th.dataset.sort));
        });
        document.getElementById('btn-eval-compare').addEventListener('click', () => this.compareResults('eval'));
    },

    renderBenchResults() {
        const state = this.resultsState.bench;
        const tbody = document.getElementById('bench-results-tbody');
        const countEl = document.getElementById('bench-results-count');

        let results = [...state.data];

        // Filter
        if (state.filter.search) {
            const term = state.filter.search.toLowerCase();
            results = results.filter(r => r.ModelName?.toLowerCase().includes(term));
        }
        if (state.filter.status) {
            results = results.filter(r => r.BenchStatus === state.filter.status);
        }

        countEl.textContent = `${results.length} results`;

        // Sort
        results.sort((a, b) => {
            const valA = a[state.sort.col];
            const valB = b[state.sort.col];
            if (valA === valB) return 0;
            if (valA === null || valA === undefined) return 1;
            if (valB === null || valB === undefined) return -1;
            return state.sort.asc ? (valA > valB ? 1 : -1) : (valA < valB ? 1 : -1);
        });

        // Update headers sort icons
        document.querySelectorAll('#bench-results-table th.sortable').forEach(th => {
            th.classList.remove('sort-asc', 'sort-desc');
            if (th.dataset.sort === state.sort.col) {
                th.classList.add(state.sort.asc ? 'sort-asc' : 'sort-desc');
            }
        });

        if (results.length === 0) {
            tbody.innerHTML = '<tr><td colspan="9" class="empty-message">No results found</td></tr>';
            return;
        }

        tbody.innerHTML = results.map((r, i) => `
            <tr class="${state.selected.has(r) ? 'selected' : ''}">
                <td class="col-checkbox"><input type="checkbox" onchange="App.handleResultsSelect('bench', ${state.data.indexOf(r)}, this.checked)" ${state.selected.has(r) ? 'checked' : ''}></td>
                <td title="${r.ModelName}">${r.ModelName}</td>
                <td>${r.ParameterSize || '-'}</td>
                <td>${r.Quantization || '-'}</td>
                <td>${r._tokens_per_second ? r._tokens_per_second.toFixed(2) : '-'}</td>
                <td>${r._gpu_memory_gb ? r._gpu_memory_gb.toFixed(2) + ' GB' : '-'}</td>
                <td>${r._duration_seconds ? r._duration_seconds.toFixed(2) + 's' : '-'}</td>
                <td><span class="status-badge ${r.BenchStatus === 'Success' ? 'status-running' : 'status-error'}">${r.BenchStatus || 'Unknown'}</span></td>
                <td title="${r.Timestamp}">${new Date(r.Timestamp).toLocaleDateString()}</td>
            </tr>
        `).join('');

        this.updateCompareButton('bench');
    },

    renderMemoryResults() {
        const state = this.resultsState.memory;
        const tbody = document.getElementById('memory-results-tbody');
        const countEl = document.getElementById('memory-results-count');

        let results = [...state.data];

        // Filter
        if (state.filter.search) {
            const term = state.filter.search.toLowerCase();
            results = results.filter(r => r.ModelName?.toLowerCase().includes(term));
        }
        if (state.filter.status) {
            results = results.filter(r => r.ScanStatus === state.filter.status);
        }

        countEl.textContent = `${results.length} results`;

        // Sort
        results.sort((a, b) => {
            const valA = a[state.sort.col];
            const valB = b[state.sort.col];
            if (valA === valB) return 0;
            if (valA === null || valA === undefined) return 1;
            if (valB === null || valB === undefined) return -1;
            return state.sort.asc ? (valA > valB ? 1 : -1) : (valA < valB ? 1 : -1);
        });

        // Update headers
        document.querySelectorAll('#memory-results-table th.sortable').forEach(th => {
            th.classList.remove('sort-asc', 'sort-desc');
            if (th.dataset.sort === state.sort.col) {
                th.classList.add(state.sort.asc ? 'sort-asc' : 'sort-desc');
            }
        });

        if (results.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="empty-message">No results found</td></tr>';
            return;
        }

        tbody.innerHTML = results.map((r, i) => `
            <tr class="${state.selected.has(r) ? 'selected' : ''}">
                <td class="col-checkbox"><input type="checkbox" onchange="App.handleResultsSelect('memory', ${state.data.indexOf(r)}, this.checked)" ${state.selected.has(r) ? 'checked' : ''}></td>
                <td title="${r.ModelName}">${r.ModelName}</td>
                <td>${r._gpu_memory_gb ? r._gpu_memory_gb.toFixed(2) + ' GB' : '-'}</td>
                <td>${r._cpu_memory_gb ? r._cpu_memory_gb.toFixed(2) + ' GB' : '-'}</td>
                <td><span class="status-badge ${r.ScanStatus === 'Success' ? 'status-running' : 'status-error'}">${r.ScanStatus || 'Unknown'}</span></td>
                <td title="${r.Timestamp}">${new Date(r.Timestamp).toLocaleDateString()}</td>
                <td class="col-error" title="${r.Error || ''}">${r.Error || ''}</td>
            </tr>
        `).join('');

        this.updateCompareButton('memory');
    },

    renderEvalResults() {
        const state = this.resultsState.eval;
        const tbody = document.getElementById('eval-results-tbody');
        const countEl = document.getElementById('eval-results-count');

        let results = [...state.data];

        // Filter
        if (state.filter.search) {
            const term = state.filter.search.toLowerCase();
            results = results.filter(r =>
                (r.ModelName?.toLowerCase().includes(term)) ||
                (r.RunName?.toLowerCase().includes(term))
            );
        }

        countEl.textContent = `${results.length} results`;

        // Sort
        results.sort((a, b) => {
            const valA = a[state.sort.col];
            const valB = b[state.sort.col];
            if (valA === valB) return 0;
            if (valA === null || valA === undefined) return 1;
            if (valB === null || valB === undefined) return -1;
            return state.sort.asc ? (valA > valB ? 1 : -1) : (valA < valB ? 1 : -1);
        });

        // Update headers
        document.querySelectorAll('#eval-results-table th.sortable').forEach(th => {
            th.classList.remove('sort-asc', 'sort-desc');
            if (th.dataset.sort === state.sort.col) {
                th.classList.add(state.sort.asc ? 'sort-asc' : 'sort-desc');
            }
        });

        if (results.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="empty-message">No results found</td></tr>';
            return;
        }

        tbody.innerHTML = results.map((r, i) => `
            <tr class="${state.selected.has(r) ? 'selected' : ''}">
                <td class="col-checkbox"><input type="checkbox" onchange="App.handleResultsSelect('eval', ${state.data.indexOf(r)}, this.checked)" ${state.selected.has(r) ? 'checked' : ''}></td>
                <td title="${r.ModelName}">${r.ModelName}</td>
                <td title="${r.RunName}">${r.RunName}</td>
                <td>${r.gen_judge_score || r.mcq_accuracy || '-'}</td>
                <td>${r.count_total || '-'}</td>
                <td>${r.tokens_total_avg ? r.tokens_total_avg.toFixed(1) : '-'}</td>
            </tr>
        `).join('');

        this.updateCompareButton('eval');
    },

    handleResultsSort(type, col) {
        const state = this.resultsState[type];
        if (state.sort.col === col) {
            state.sort.asc = !state.sort.asc;
        } else {
            state.sort.col = col;
            state.sort.asc = true; // Default to asc for new col
            // Special defaults for numeric metrics - desc usually better
            if (['_tokens_per_second', 'gen_judge_score', 'count_total'].includes(col)) {
                state.sort.asc = false;
            }
        }
        if (type === 'bench') this.renderBenchResults();
        if (type === 'memory') this.renderMemoryResults();
        if (type === 'eval') this.renderEvalResults();
    },

    handleResultsSelect(type, index, checked) {
        const state = this.resultsState[type];
        const item = state.data[index];
        if (checked) {
            state.selected.add(item);
        } else {
            state.selected.delete(item);
        }
        this.updateCompareButton(type);
        // Re-render just to update row highlighting? No need for full re-render, but it's cleaner
        if (type === 'bench') this.renderBenchResults();
        if (type === 'memory') this.renderMemoryResults();
        if (type === 'eval') this.renderEvalResults();
    },

    toggleSelectAll(type, checked) {
        const state = this.resultsState[type];
        if (checked) {
            // Add all currently visible items (respecting filters)
            // Ideally we need to know which are visible. Simple way: data is filtered in render
            // Re-filtering here is duplication.
            // Simplified: Add ALL items from data
            state.data.forEach(item => state.selected.add(item));
        } else {
            state.selected.clear();
        }
        if (type === 'bench') this.renderBenchResults();
        if (type === 'memory') this.renderMemoryResults();
        if (type === 'eval') this.renderEvalResults();
    },

    updateCompareButton(type) {
        const state = this.resultsState[type];
        const btn = document.getElementById(`btn-${type}-compare`);
        const countSpan = document.getElementById(`${type}-selected-count`);
        const count = state.selected.size;

        countSpan.textContent = count;
        btn.disabled = count < 2;
    },

    compareResults(type) {
        const state = this.resultsState[type];
        const selected = Array.from(state.selected);

        if (selected.length < 2) return;

        let content = '';
        if (type === 'bench') {
            content = `
                <div class="table-container">
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                ${selected.map(r => `<th>${r.ModelName.substring(0, 20)}...</th>`).join('')}
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td><strong>Tokens/Sec</strong></td>${selected.map(r => `<td>${r._tokens_per_second?.toFixed(2) || '-'}</td>`).join('')}</tr>
                            <tr><td><strong>GPU Memory</strong></td>${selected.map(r => `<td>${r._gpu_memory_gb?.toFixed(2) || '-'} GB</td>`).join('')}</tr>
                            <tr><td><strong>Duration</strong></td>${selected.map(r => `<td>${r._duration_seconds?.toFixed(2) || '-'} s</td>`).join('')}</tr>
                            <tr><td><strong>Context</strong></td>${selected.map(r => `<td>${r.ContextSize || '-'}</td>`).join('')}</tr>
                            <tr><td><strong>Quantization</strong></td>${selected.map(r => `<td>${r.Quantization || '-'}</td>`).join('')}</tr>
                            <tr><td><strong>Params</strong></td>${selected.map(r => `<td>${r.ParameterSize || '-'}</td>`).join('')}</tr>
                        </tbody>
                    </table>
                </div>`;
        } else if (type === 'memory') {
            content = `
                <div class="table-container">
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                ${selected.map(r => `<th>${r.ModelName.substring(0, 20)}...</th>`).join('')}
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td><strong>GPU Memory</strong></td>${selected.map(r => `<td>${r._gpu_memory_gb?.toFixed(2) || '-'} GB</td>`).join('')}</tr>
                            <tr><td><strong>CPU Memory</strong></td>${selected.map(r => `<td>${r._cpu_memory_gb?.toFixed(2) || '-'} GB</td>`).join('')}</tr>
                            <tr><td><strong>Status</strong></td>${selected.map(r => `<td>${r.ScanStatus || '-'}</td>`).join('')}</tr>
                        </tbody>
                    </table>
                </div>`;
        } else if (type === 'eval') {
            content = `
                <div class="table-container">
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                ${selected.map(r => `<th>${r.ModelName.substring(0, 20)}...</th>`).join('')}
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td><strong>Judge Score</strong></td>${selected.map(r => `<td>${r.gen_judge_score || '-'}</td>`).join('')}</tr>
                            <tr><td><strong>MCQ Acc</strong></td>${selected.map(r => `<td>${r.mcq_accuracy || '-'}</td>`).join('')}</tr>
                            <tr><td><strong>Tasks</strong></td>${selected.map(r => `<td>${r.count_total || '-'}</td>`).join('')}</tr>
                            <tr><td><strong>Avg Tokens</strong></td>${selected.map(r => `<td>${r.tokens_total_avg?.toFixed(1) || '-'}</td>`).join('')}</tr>
                        </tbody>
                    </table>
                </div>`;
        }

        Modal.show('Compare Results', content, '<button class="btn btn-secondary" onclick="Modal.hide()">Close</button>');
    },

    // Action handlers
    async saveConfig() {
        try {
            const content = document.getElementById('config-editor').value;
            if ((this.baseConfigOriginal ?? '') === content) {
                Toast.info('No changes to save');
                return;
            }
            await API.put('/api/config', { content });
            Toast.success('Configuration saved');
            this.baseConfigOriginal = content;
            this.updateBaseConfigSaveButton();
            await this.loadEffectiveConfig();
        } catch (e) {
            Toast.error(`Failed to save: ${e.message}`);
        }
    },

    async copyEffectiveConfig() {
        const text = document.getElementById('effective-config')?.textContent || '';
        if (!text.trim()) {
            Toast.warning('Nothing to copy');
            return;
        }

        try {
            await navigator.clipboard.writeText(text);
            Toast.success('Copied effective config');
        } catch {
            // Fallback for older browsers / blocked clipboard APIs
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.left = '-9999px';
            document.body.appendChild(ta);
            ta.select();
            try {
                document.execCommand('copy');
                Toast.success('Copied effective config');
            } catch {
                Toast.error('Copy failed');
            } finally {
                ta.remove();
            }
        }
    },

    editCurrentOverride() {
        if (!this.currentOverride) {
            Toast.info('Select an override first');
            return;
        }
        this.editOverride(this.currentOverride);
    },

    async duplicateCurrentOverride() {
        if (!this.currentOverride) {
            Toast.info('Select an override first');
            return;
        }

        try {
            const data = await API.get(`/api/config/overrides/${this.currentOverride}`);
            const suggested = `${this.currentOverride}-copy`;
            this.showNewOverrideDialog({ name: suggested, content: data.content });
        } catch (e) {
            Toast.error(`Failed to duplicate: ${e.message}`);
        }
    },

    showNewOverrideDialog(preset = {}) {
        Modal.show('New Override', `
            <div class="form-grid" style="grid-template-columns: 1fr;">
                <div class="form-group">
                    <label for="new-override-name">Name</label>
                    <input type="text" id="new-override-name" placeholder="e.g., win-5090" value="">
                    <div class="muted" style="margin-top: 0.25rem;">Creates <code>configs/overrides/&lt;name&gt;.yaml</code></div>
                </div>
                <div class="form-group">
                    <label for="new-override-content">YAML</label>
                    <textarea id="new-override-content" class="code-editor" style="min-height: 260px;"></textarea>
                </div>
            </div>
        `, `
            <button class="btn btn-secondary" onclick="Modal.hide()">Cancel</button>
            <button class="btn btn-primary" onclick="App.createOverrideFromDialog()">Create</button>
        `);

        const defaultName = preset.name || '';
        const defaultContent = preset.content || this.buildOverrideTemplate();
        const nameEl = document.getElementById('new-override-name');
        const contentEl = document.getElementById('new-override-content');
        if (nameEl) nameEl.value = defaultName;
        if (contentEl) contentEl.value = defaultContent;
    },

    buildOverrideTemplate() {
        return [
            '# Machine override',
            '# Keep this file small: only put machine-specific changes here.',
            '# Example:',
            '#   models:',
            '#     SomeModelName:',
            '#       cmd:',
            '#         gpu-layers: 99',
            '',
        ].join('\\n');
    },

    async createOverrideFromDialog() {
        const name = (document.getElementById('new-override-name')?.value || '').trim();
        const content = document.getElementById('new-override-content')?.value || '';

        if (!name) {
            Toast.warning('Override name is required');
            return;
        }

        try {
            const created = await API.post('/api/config/overrides', { name, content });
            Toast.success(`Created override: ${created.name}`);
            Modal.hide();
            await this.loadOverrides();
            await this.loadConfig();

            // Select the newly created override and persist it
            this.currentOverride = created.name;
            const select = document.getElementById('current-override');
            if (select) select.value = created.name;
            try {
                localStorage.setItem('llama-suite.override', created.name);
            } catch {
                // ignore
            }
            await this.refreshCurrentSection();
        } catch (e) {
            Toast.error(`Failed to create override: ${e.message}`);
        }
    },

    async editOverride(name) {
        try {
            const data = await API.get(`/api/config/overrides/${name}`);
            Modal.show(`Edit Override: ${name}`, `
                <div class="form-group">
                    <textarea id="override-editor" class="code-editor" style="min-height: 300px;">${data.content}</textarea>
                </div>
            `, `
                <button class="btn btn-secondary" onclick="Modal.hide()">Cancel</button>
                <button class="btn btn-primary" onclick="App.saveOverride('${name}')">Save</button>
            `);
        } catch (e) {
            Toast.error('Failed to load override');
        }
    },

    async saveOverride(name) {
        try {
            const content = document.getElementById('override-editor').value;
            await API.put(`/api/config/overrides/${name}`, { content });
            Toast.success('Override saved');
            Modal.hide();
            await this.loadConfig();
        } catch (e) {
            Toast.error(`Failed to save: ${e.message}`);
        }
    },

    async deleteOverride(name) {
        if (!confirm(`Delete override "${name}"?`)) return;
        try {
            await API.delete(`/api/config/overrides/${name}`);
            Toast.success('Override deleted');
            await this.loadConfig();
            await this.loadOverrides();
            await this.refreshCurrentSection();
        } catch (e) {
            Toast.error('Failed to delete override');
        }
    },

    async runBenchmark() {
        const output = document.getElementById('bench-output');
        output.innerHTML = '';

        try {
            const data = await API.post('/api/bench/run', {
                override: this.currentOverride || undefined,
                model: document.getElementById('bench-model').value || undefined,
                question: document.getElementById('bench-question').value,
                health_timeout: parseInt(document.getElementById('bench-timeout').value)
            });
            Toast.info(`Benchmark started: ${data.task_id}`);
            this.currentBenchTaskId = data.task_id;
            this.setTaskRunningState('bench', true);

            // Add to Task Manager
            this.taskManager.addTask(data.task_id, 'bench', 'Benchmark: ' + (document.getElementById('bench-model').value || 'Default Model'));

        } catch (e) {
            Toast.error(`Failed to start: ${e.message}`);
        }
    },

    async stopBenchmark() {
        if (this.currentBenchTaskId) {
            await this.cancelTask(this.currentBenchTaskId);
            // State reset handled by ws complete/cancel event or here optimistically?
            // Let's do it here optimistically to feel responsive
            this.setTaskRunningState('bench', false);
            this.currentBenchTaskId = null;
        }
    },

    async runMemoryScan() {
        const output = document.getElementById('memory-output');
        output.innerHTML = '';

        try {
            const data = await API.post('/api/memory/run', {
                override: this.currentOverride || undefined,
                model: document.getElementById('memory-model').value || undefined,
                health_timeout: parseInt(document.getElementById('memory-timeout').value)
            });
            Toast.info(`Memory scan started: ${data.task_id}`);
            this.currentMemoryTaskId = data.task_id;
            this.setTaskRunningState('memory', true);

            // Add to Task Manager
            this.taskManager.addTask(data.task_id, 'memory', 'Memory Scan: ' + (document.getElementById('memory-model').value || 'Default Model'));

        } catch (e) {
            Toast.error(`Failed to start: ${e.message}`);
        }
    },

    async stopMemoryScan() {
        if (this.currentMemoryTaskId) {
            await this.cancelTask(this.currentMemoryTaskId);
            this.setTaskRunningState('memory', false);
            this.currentMemoryTaskId = null;
        }
    },

    async runEvalHarness() {
        const output = document.getElementById('eval-output');
        output.innerHTML = '';

        try {
            const tasks = document.getElementById('eval-tasks').value;
            const data = await API.post('/api/eval/harness/run', {
                override: this.currentOverride || undefined,
                model: document.getElementById('eval-model').value || undefined,
                tasks: tasks,
                limit: document.getElementById('eval-limit').value ? parseFloat(document.getElementById('eval-limit').value) : undefined,
                num_fewshot: document.getElementById('eval-fewshot').value ? parseInt(document.getElementById('eval-fewshot').value) : undefined,
                batch_size: document.getElementById('eval-batch').value
            });
            Toast.info(`Eval-harness started: ${data.task_id}`);
            this.currentEvalHarnessTaskId = data.task_id;
            this.setTaskRunningState('eval-harness', true);

            // Add to Task Manager
            this.taskManager.addTask(data.task_id, 'eval', `Eval Harness: ${tasks}`);

        } catch (e) {
            Toast.error(`Failed to start: ${e.message}`);
        }
    },

    async stopEvalHarness() {
        if (this.currentEvalHarnessTaskId) {
            await this.cancelTask(this.currentEvalHarnessTaskId);
            this.setTaskRunningState('eval-harness', false);
            this.currentEvalHarnessTaskId = null;
        }
    },

    async runCustomEval() {
        const output = document.getElementById('eval-output');
        output.innerHTML = '';

        try {
            const dataset = document.getElementById('custom-dataset').value;
            const data = await API.post('/api/eval/custom/run', {
                override: this.currentOverride || undefined,
                model: document.getElementById('custom-model').value || undefined,
                dataset: dataset,
                max_tasks: document.getElementById('custom-max-tasks').value ? parseInt(document.getElementById('custom-max-tasks').value) : undefined,
                temperature: document.getElementById('custom-temp').value ? parseFloat(document.getElementById('custom-temp').value) : undefined
            });
            Toast.info(`Custom eval started: ${data.task_id}`);
            this.currentEvalCustomTaskId = data.task_id;
            this.setTaskRunningState('eval-custom', true);

            // Add to Task Manager
            this.taskManager.addTask(data.task_id, 'eval', `Custom Eval: ${dataset}`);

        } catch (e) {
            Toast.error(`Failed to start: ${e.message}`);
        }
    },

    async stopCustomEval() {
        if (this.currentEvalCustomTaskId) {
            await this.cancelTask(this.currentEvalCustomTaskId);
            this.setTaskRunningState('eval-custom', false);
            this.currentEvalCustomTaskId = null;
        }
    },

    async startWatcher() {
        const output = document.getElementById('watcher-output');
        output.innerHTML = '';

        try {
            const data = await API.post('/api/watcher/start', {
                override: this.currentOverride || undefined,
                verbose: document.getElementById('watcher-verbose').checked,
                dry_run: document.getElementById('watcher-dry-run').checked
            });

            // Register task container for log routing
            this.registerTaskContainer(data.task_id, 'watcher-output');
            this.currentWatcherTaskId = data.task_id;

            Toast.info(`Endpoint started: ${data.task_id}`);
            await this.loadWatcherStatus();
            // Watcher button state is handled by checkWatcherStatus/loadWatcherStatus usually,
            // but we can set it here too.
            this.setWatcherRunningState(true);
        } catch (e) {
            Toast.error(`Failed to start endpoint: ${e.message}`);
        }
    },

    async stopWatcher() {
        try {
            await API.post('/api/watcher/stop');

            // Unregister task container
            if (this.currentWatcherTaskId) {
                this.unregisterTaskContainer(this.currentWatcherTaskId);
                this.currentWatcherTaskId = null;
            }

            Toast.success('Endpoint stopped');
            await this.loadWatcherStatus();
            this.setWatcherRunningState(false);
        } catch (e) {
            Toast.error('Failed to stop endpoint');
        }
    },

    async startOpenWebUI(portOverride = null) {
        const output = document.getElementById('system-output');
        if (output) output.innerHTML = '';

        try {
            const port = portOverride ?? parseInt(document.getElementById('openwebui-port')?.value || '3000', 10);
            const dataVolume = (document.getElementById('openwebui-data-volume')?.value || '').trim();
            const payload = { port };
            if (dataVolume) payload.data_volume = dataVolume;
            const data = await API.post('/api/system/openwebui/start', payload);

            this.taskManager.addTask(data.task_id, 'system', 'Open WebUI container');
            this.registerTaskContainer(data.task_id, 'system-output');

            Toast.info(`Open WebUI start requested: ${data.task_id}`);
            this.refreshDashboard();
        } catch (e) {
            Toast.error(`Failed to start Open WebUI: ${e.message}`);
        }
    },

    async stopOpenWebUI() {
        const output = document.getElementById('system-output');
        if (output) output.innerHTML = '';

        try {
            const data = await API.post('/api/system/openwebui/stop', {});

            this.taskManager.addTask(data.task_id, 'system', 'Open WebUI container (stop)');
            this.registerTaskContainer(data.task_id, 'system-output');

            Toast.info(`Open WebUI stop requested: ${data.task_id}`);
            this.refreshDashboard();
        } catch (e) {
            Toast.error(`Failed to stop Open WebUI: ${e.message}`);
        }
    },

    async runUpdate() {
        const output = document.getElementById('system-output');
        if (output) output.innerHTML = '';

        try {
            const dataVolume = (document.getElementById('openwebui-data-volume')?.value || '').trim();
            const data = await API.post('/api/system/update', {
                update_python: document.getElementById('update-python').checked,
                update_llama_swap: document.getElementById('update-swap').checked,
                update_llama_cpp: document.getElementById('update-cpp').checked,
                update_open_webui: document.getElementById('update-webui').checked,
                open_webui_data_volume: dataVolume || undefined,
                gpu_backend: document.getElementById('update-gpu').value
            });

            // Show task immediately and route logs to the System output panel
            this.taskManager.addTask(data.task_id, 'system', 'Update Components');
            this.registerTaskContainer(data.task_id, 'system-output');

            Toast.info(`Update started: ${data.task_id}`);
            this.refreshDashboard();
        } catch (e) {
            Toast.error(`Failed to start: ${e.message}`);
        }
    },

    async downloadModels() {
        const output = document.getElementById('system-output');
        if (output) output.innerHTML = '';

        try {
            const force = document.getElementById('download-force').checked;
            const selected = Array.from(document.querySelectorAll('.download-model-checkbox:checked')).map(el => el.value);
            const hasMissingList = this.missingModelsForDownload && this.missingModelsForDownload.length > 0;

            if (hasMissingList && selected.length === 0) {
                Toast.warning('Select at least one missing model to download');
                return;
            }

            const data = await API.post('/api/system/download', {
                override: this.currentOverride || undefined,
                models: selected.length > 0 ? selected : undefined,
                include_drafts: document.getElementById('download-drafts').checked,
                include_tokenizers: document.getElementById('download-tokenizers').checked,
                force
            });

            // Show task immediately and route logs to the System output panel
            const label = selected.length > 0 ? `Download ${selected.length} model(s)` : 'Download Models';
            this.taskManager.addTask(data.task_id, 'system', label);
            this.registerTaskContainer(data.task_id, 'system-output');

            Toast.info(`Download started: ${data.task_id}`);
            this.refreshDashboard();
        } catch (e) {
            Toast.error(`Failed to start: ${e.message}`);
        }
    },

    async cancelTask(taskId) {
        try {
            await API.post(`/api/system/cancel/${taskId}`);
            Toast.success('Task cancelled');
            this.taskManager.cancelTaskUI(taskId);
            await this.refreshDashboard();
        } catch (e) {
            Toast.error('Failed to cancel task');
        }
    },

    async viewResult(type, name) {
        try {
            const data = await API.get(`/api/results/${type}/${name}`);
            const content = Object.entries(data.data).map(([file, content]) => `
                <h4>${file}</h4>
                <pre class="code-display" style="max-height: 300px; overflow: auto;">${typeof content === 'object' ? JSON.stringify(content, null, 2) : content
                }</pre>
            `).join('');
            Modal.show(`Results: ${name}`, content || '<p>No data files found</p>');
        } catch (e) {
            Toast.error('Failed to load results');
        }
    },

    async deleteResult(type, name) {
        if (!confirm(`Delete ${type} result "${name}"?`)) return;
        try {
            await API.delete(`/api/results/${type}/${name}`);
            Toast.success('Result deleted');
            await this.loadResults();
            await this.refreshDashboard();
        } catch (e) {
            Toast.error('Failed to delete result');
        }
    },

    async showAddModelDialog(preset = null) {
        try {
            // Load available GGUF files
            const filesData = await API.get('/api/models/files/available');
            const files = filesData.files || [];

            // Load existing models for copy-from option
            const modelsData = await API.get('/api/models');
            const existingModels = modelsData.models || [];

            Modal.show('Add New Model', `
                <div class="form-grid" style="grid-template-columns: 1fr;">
                    <div class="muted" style="margin-bottom: 0.5rem;">
                        This creates a config entry. It does not download or upload files.
                        Use <strong>Upload local GGUF</strong> if you already have a file, or <strong>Get missing models</strong> to download from Hugging Face.
                        <div style="display:flex; gap: 0.5rem; margin-top: 0.5rem; flex-wrap: wrap;">
                            <button type="button" class="btn btn-secondary btn-sm" onclick="App.showUploadDialog()">Upload local GGUF…</button>
                            <button type="button" class="btn btn-secondary btn-sm" onclick="App.goToDownloadModels()">Get missing models…</button>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Model Name (alias)</label>
                        <input type="text" id="new-model-name" placeholder="e.g., qwen3-8b-q4" required>
                    </div>
                    <div class="form-group">
                        <label>Model File</label>
                        <select id="new-model-file">
                            <option value="">-- Select a GGUF file --</option>
                            ${files.map(f => `<option value="${f.config_path}">${f.relative_path || f.name} (${(f.size_bytes / 1e9).toFixed(2)} GB)</option>`).join('')}
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Or enter model path manually</label>
                        <input type="text" id="new-model-path" placeholder="./models/your-model.gguf">
                    </div>
                    <div class="form-group">
                        <label>Context Size</label>
                        <input type="number" id="new-ctx-size" value="8192">
                    </div>
                    <div class="form-group">
                        <label>GPU Layers (-1 = all)</label>
                        <input type="number" id="new-gpu-layers" value="-1">
                    </div>
                    <div class="form-group">
                        <label>HF Tokenizer (optional)</label>
                        <input type="text" id="new-tokenizer" placeholder="e.g., Qwen/Qwen3-8B">
                    </div>
                    <div class="form-group">
                        <label>Copy settings from existing model (optional)</label>
                        <select id="new-copy-from">
                            <option value="">-- None --</option>
                            ${existingModels.map(m => `<option value="${m.name}">${m.name}</option>`).join('')}
                        </select>
                    </div>
                </div>
            `, `
                <button class="btn btn-secondary" onclick="Modal.hide()">Cancel</button>
                <button class="btn btn-primary" onclick="App.createModel()">Create Model</button>
            `);

            if (preset && typeof preset === 'object') {
                const nameEl = document.getElementById('new-model-name');
                const pathEl = document.getElementById('new-model-path');
                const fileEl = document.getElementById('new-model-file');
                if (preset.name && nameEl) nameEl.value = String(preset.name);
                if (preset.model_path) {
                    const p = String(preset.model_path);
                    if (fileEl && Array.from(fileEl.options).some(o => o.value === p)) {
                        fileEl.value = p;
                    } else if (pathEl) {
                        pathEl.value = p;
                    }
                }
            }
        } catch (e) {
            Toast.error(`Failed to load data: ${e.message}`);
        }
    },

    async createModel() {
        try {
            const name = document.getElementById('new-model-name').value.trim();
            if (!name) {
                Toast.warning('Please enter a model name');
                return;
            }

            const fileSelect = document.getElementById('new-model-file').value;
            const manualPath = document.getElementById('new-model-path').value.trim();
            const modelPath = fileSelect || manualPath;

            if (!modelPath) {
                Toast.warning('Please select or enter a model path');
                return;
            }

            const ctxSize = parseInt(document.getElementById('new-ctx-size').value) || 8192;
            const gpuLayers = parseInt(document.getElementById('new-gpu-layers').value);
            const tokenizer = document.getElementById('new-tokenizer').value.trim();
            const copyFrom = document.getElementById('new-copy-from').value;

            await API.post(`/api/models?name=${encodeURIComponent(name)}`, {
                model_path: modelPath,
                ctx_size: ctxSize,
                gpu_layers: isNaN(gpuLayers) ? -1 : gpuLayers,
                hf_tokenizer: tokenizer || undefined,
                copy_from: copyFrom || undefined
            });

            Toast.success(`Model "${name}" created`);
            Modal.hide();
            await this.loadModels();
            await this.refreshDashboard();
        } catch (e) {
            Toast.error(`Failed to create: ${e.message}`);
        }
    },

    async showUploadDialog() {
        Modal.show('Upload local GGUF (from your computer)', `
            <div class="form-grid" style="grid-template-columns: 1fr;">
                <div class="muted" style="margin-bottom: 0.5rem;">
                    This copies a GGUF file into your <code>models/</code> folder. It does <strong>not</strong> download from the internet.
                </div>
                <div class="form-group">
                    <label>Select GGUF file</label>
                    <input type="file" id="upload-file" accept=".gguf" style="padding: 1rem; border: 2px dashed var(--border-light); border-radius: var(--radius-md); width: 100%;">
                </div>
                <div class="form-group">
                    <label>Subfolder (optional)</label>
                    <input type="text" id="upload-subfolder" placeholder="e.g., qwen or leave empty for root">
                </div>
                <div class="form-group">
                    <label class="checkbox-label">
                        <input type="checkbox" id="upload-open-add-model" checked>
                        After upload, create a model entry
                    </label>
                </div>
                <div id="upload-progress" style="display: none;">
                    <div class="task-item">
                        <div class="task-spinner"></div>
                        <div class="task-info">
                            <div class="task-name">Uploading...</div>
                            <div class="task-progress" id="upload-status">0%</div>
                        </div>
                    </div>
                </div>
            </div>
        `, `
            <button class="btn btn-secondary" onclick="Modal.hide()">Cancel</button>
            <button class="btn btn-primary" id="btn-upload" onclick="App.uploadModel()">Upload</button>
        `);
    },

    async uploadModel() {
        const fileInput = document.getElementById('upload-file');
        const file = fileInput.files[0];

        if (!file) {
            Toast.warning('Please select a file');
            return;
        }

        if (!file.name.endsWith('.gguf')) {
            Toast.error('Only .gguf files are allowed');
            return;
        }

        const subfolder = document.getElementById('upload-subfolder').value.trim();
        const progressDiv = document.getElementById('upload-progress');
        const statusDiv = document.getElementById('upload-status');
        const uploadBtn = document.getElementById('btn-upload');
        const openAddModel = document.getElementById('upload-open-add-model')?.checked ?? true;

        progressDiv.style.display = 'block';
        uploadBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('subfolder', subfolder);

        try {
            const response = await fetch('/api/models/files/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'Upload failed');
            }

            Toast.success(`Uploaded: ${result.filename} (${(result.size_bytes / 1e9).toFixed(2)} GB)`);
            Modal.hide();
            await this.loadModels();
            await this.refreshDashboard();

            if (openAddModel) {
                const suggestedName = (result.filename || '').replace(/\.gguf$/i, '');
                const configPath = result.config_path || '';
                await this.showAddModelDialog({
                    name: suggestedName || undefined,
                    model_path: configPath || undefined
                });
            }
        } catch (e) {
            Toast.error(`Upload failed: ${e.message}`);
            progressDiv.style.display = 'none';
            uploadBtn.disabled = false;
        }
    }
};

// =============================================================================
// Event Bindings
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    App.init();

    const bind = (id, event, handler) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener(event, handler);
    };

    // Header: overrides
    bind('btn-override-new', 'click', () => App.showNewOverrideDialog());
    bind('btn-override-edit', 'click', () => App.editCurrentOverride());
    bind('btn-override-duplicate', 'click', () => App.duplicateCurrentOverride());

    bind('btn-save-config', 'click', () => App.saveConfig());
    bind('btn-new-override', 'click', () => App.showNewOverrideDialog());
    bind('btn-copy-effective-config', 'click', () => App.copyEffectiveConfig());

    bind('bench-form', 'submit', (e) => {
        e.preventDefault();
        App.runBenchmark();
    });

    bind('memory-form', 'submit', (e) => {
        e.preventDefault();
        App.runMemoryScan();
    });

    bind('eval-harness-form', 'submit', (e) => {
        e.preventDefault();
        App.runEvalHarness();
    });

    bind('eval-custom-form', 'submit', (e) => {
        e.preventDefault();
        App.runCustomEval();
    });

    bind('watcher-form', 'submit', (e) => {
        e.preventDefault();
        App.startWatcher();
    });
    bind('btn-watcher-stop', 'click', () => App.stopWatcher());

    // Dashboard Quick Start
    bind('btn-quickstart-endpoint-start', 'click', () => App.startWatcher());
    bind('btn-quickstart-endpoint-stop', 'click', () => App.stopWatcher());
    bind('btn-quickstart-openwebui-start', 'click', () => App.startOpenWebUI());
    bind('btn-quickstart-openwebui-stop', 'click', () => App.stopOpenWebUI());

    // System: Open WebUI
    bind('openwebui-form', 'submit', (e) => {
        e.preventDefault();
        App.startOpenWebUI();
    });
    bind('btn-openwebui-stop', 'click', () => App.stopOpenWebUI());

    bind('update-form', 'submit', (e) => {
        e.preventDefault();
        App.runUpdate();
    });

    bind('download-form', 'submit', (e) => {
        e.preventDefault();
        App.downloadModels();
    });
    bind('download-force', 'change', () => App.loadDownloadModelsUI());
});

// Make App globally available for inline handlers
window.App = App;
window.runBenchmark = () => App.runBenchmark();
window.runMemoryScan = () => App.runMemoryScan();
window.runEvalHarness = () => App.runEvalHarness();
window.runCustomEval = () => App.runCustomEval();
window.cancelTask = (id) => App.cancelTask(id);
window.stopBenchmark = () => App.stopBenchmark();
window.stopMemoryScan = () => App.stopMemoryScan();
window.stopEvalHarness = () => App.stopEvalHarness();
window.stopCustomEval = () => App.stopCustomEval();
