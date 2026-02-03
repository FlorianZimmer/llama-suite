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
        const retryAuth = options._retryAuth !== false;
        // Internal option, not forwarded to fetch()
        if ("_retryAuth" in options) {
            const { _retryAuth, ...rest } = options;
            options = rest;
        }
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
            const ct = (response.headers.get('content-type') || '').toLowerCase();
            const isJson = ct.includes('application/json');
            const data = isJson ? await response.json() : await response.text();

            if (response.status === 401 && retryAuth && endpoint.startsWith('/api/') && !endpoint.startsWith('/api/auth/')) {
                try {
                    await Auth.ensureLoggedIn();
                    return await this.request(endpoint, { ...options, _retryAuth: false });
                } catch (e) {
                    // Fall through to normal error handling below
                }
            }

            if (!response.ok) {
                const detail = (data && typeof data === 'object' && data.detail) ? data.detail : null;
                throw new Error(detail || `HTTP ${response.status}`);
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
// Auth helper (optional API key + cookie)
// =============================================================================

const Auth = {
    _inFlight: null,

    async status() {
        try {
            const resp = await fetch('/api/auth/status', { method: 'GET' });
            if (!resp.ok) return { enabled: false, authenticated: true };
            return await resp.json();
        } catch {
            return { enabled: false, authenticated: true };
        }
    },

    async login(apiKey) {
        const resp = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: apiKey })
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
            throw new Error(data.detail || `HTTP ${resp.status}`);
        }
        return true;
    },

    promptForKey() {
        return new Promise((resolve) => {
            Modal.show('Login required', `
                <div class="form-grid" style="grid-template-columns: 1fr;">
                    <div class="form-group">
                        <label for="auth-api-key">API key</label>
                        <input type="password" id="auth-api-key" placeholder="Enter LLAMA_SUITE_API_KEY">
                    </div>
                    <div class="muted">This UI is protected by an API key. A session cookie will be set after login.</div>
                </div>
            `, `
                <button class="btn btn-secondary" id="btn-auth-cancel">Cancel</button>
                <button class="btn btn-primary" id="btn-auth-login">Login</button>
            `);

            const keyEl = document.getElementById('auth-api-key');
            const loginBtn = document.getElementById('btn-auth-login');
            const cancelBtn = document.getElementById('btn-auth-cancel');

            const cleanup = () => {
                loginBtn?.removeEventListener('click', onLogin);
                cancelBtn?.removeEventListener('click', onCancel);
                keyEl?.removeEventListener('keydown', onKeyDown);
            };

            const onLogin = () => {
                const key = (keyEl?.value || '').trim();
                cleanup();
                resolve(key || null);
            };

            const onCancel = () => {
                cleanup();
                resolve(null);
            };

            const onKeyDown = (e) => {
                if (e.key === 'Enter') onLogin();
            };

            loginBtn?.addEventListener('click', onLogin);
            cancelBtn?.addEventListener('click', () => { Modal.hide(); onCancel(); });
            keyEl?.addEventListener('keydown', onKeyDown);

            setTimeout(() => keyEl?.focus(), 0);
        });
    },

    async ensureLoggedIn() {
        if (this._inFlight) return this._inFlight;

        this._inFlight = (async () => {
            const st = await this.status();
            if (!st.enabled || st.authenticated) return;

            while (true) {
                const apiKey = await this.promptForKey();
                if (!apiKey) {
                    throw new Error('Login required');
                }

                try {
                    await this.login(apiKey);
                    Modal.hide();
                    Toast.success('Logged in');
                    return;
                } catch (e) {
                    Toast.error(`Login failed: ${e.message}`);
                }
            }
        })();

        try {
            return await this._inFlight;
        } finally {
            this._inFlight = null;
        }
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
    modalEl: null,
    titleEl: null,
    bodyEl: null,
    footerEl: null,

    init() {
        this.overlay = document.getElementById('modal-overlay');
        this.modalEl = document.getElementById('modal');
        this.titleEl = document.getElementById('modal-title');
        this.bodyEl = document.getElementById('modal-body');
        this.footerEl = document.getElementById('modal-footer');

        document.getElementById('modal-close').addEventListener('click', () => this.hide());
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) this.hide();
        });
    },

    setSize(size) {
        if (!this.modalEl) return;

        this.modalEl.classList.remove('modal-wide');
        if (size === 'wide') this.modalEl.classList.add('modal-wide');
    },

    show(title, bodyHtml, footerHtml = '', options = {}) {
        this.setSize(options?.size);
        this.titleEl.textContent = title;
        this.bodyEl.innerHTML = bodyHtml;
        this.footerEl.innerHTML = footerHtml;
        this.overlay.classList.add('active');
    },

    hide() {
        this.overlay.classList.remove('active');
        this.setSize(null);
    }
};

// =============================================================================
// App State
// =============================================================================

const App = {
    currentSection: 'dashboard',
    currentOverride: '',
    links: null,
    capabilities: null,
    ws: null,
    baseConfigOriginal: null,
    missingModelsForDownload: [],
    outputContainers: {},
    currentBenchTaskId: null,
    currentBenchSweepTaskId: null,
    currentMemoryTaskId: null,
    currentMemorySweepTaskId: null,
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
	        try {
	            await Auth.ensureLoggedIn();
	        } catch (e) {
	            console.warn('Auth not completed:', e);
	        }
	        this.ws = new WebSocketManager();
	
	        this.setupNavigation();
	        this.setupTabs();
	        this.setupSweepsUI();
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
	            this.updateSectionProgressFromEvent(data.task_id, progress, message);
	        });

        this.ws.on('complete', (data) => {
            const ok = !!data.success;
            (ok ? Toast.success : Toast.error)(ok ? 'Task completed' : 'Task failed');
            this.refreshDashboard();

            this.taskManager.completeTask(data.task_id, data.success, data.success ? 'Completed' : 'Failed');

            // Sweep results (best-effort)
            if (data.task_id === this.currentBenchSweepTaskId) {
                this.loadSweepResults(data.task_id, 'bench-sweep-results-table').catch(() => {});
            } else if (data.task_id === this.currentMemorySweepTaskId) {
                this.loadSweepResults(data.task_id, 'memory-sweep-results-table').catch(() => {});
            }

            this.checkAndResetTaskState(data.task_id);
        });

	        this.ws.on('cancelled', (data) => {
	            Toast.info('Task cancelled');
	            this.refreshDashboard();

	            this.taskManager.cancelTaskUI(data.task_id);

	            this.checkAndResetTaskState(data.task_id);
	        });
	    },

	    updateSectionProgressUI(sectionKey, pct, message) {
	        const fill = document.getElementById(`${sectionKey}-progress-fill`);
	        const text = document.getElementById(`${sectionKey}-progress-text`);

	        if (text) text.textContent = message || '';

	        if (fill) {
	            if (pct === -1) {
	                fill.style.width = '15%';
	            } else {
	                const clamped = Math.max(0, Math.min(100, pct));
	                fill.style.width = `${clamped}%`;
	            }
	        }
	    },

	    updateSectionProgressFromEvent(taskId, pct, message) {
	        if (taskId === this.currentMemoryTaskId) {
	            this.updateSectionProgressUI('memory', pct, message || 'Running...');
	        } else if (taskId === this.currentMemorySweepTaskId) {
	            this.updateSectionProgressUI('memory-sweep', pct, message || 'Running...');
	        } else if (taskId === this.currentEvalHarnessTaskId) {
	            this.updateSectionProgressUI('eval-harness', pct, message || 'Running...');
	        } else if (taskId === this.currentEvalCustomTaskId) {
	            this.updateSectionProgressUI('eval-custom', pct, message || 'Running...');
	        }
	    },

	    updateEndpointStatusUI(isRunning, statusTextOverride = null) {
	        const statusText = statusTextOverride || (isRunning ? 'Running' : 'Stopped');
	        const canSpawn = this.capabilities?.can_spawn_subprocesses !== false;

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
	        if (startBtn) startBtn.disabled = !canSpawn || isRunning;
	        if (stopBtn) stopBtn.disabled = !canSpawn || !isRunning;

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
	        if (qsStart) qsStart.disabled = !canSpawn || isRunning;
	        if (qsStop) qsStop.disabled = !canSpawn || !isRunning;

        if (this.taskManager) {
            this.taskManager.updateWatcherState(isRunning);
        }
    },

	    updateOpenWebUIStatusUI(isRunning, statusTextOverride = null) {
	        const statusText = statusTextOverride || (isRunning ? 'Running' : 'Stopped');
	        const cls = statusTextOverride === 'Error'
	            ? 'status-badge status-error'
	            : `status-badge ${isRunning ? 'status-running' : 'status-stopped'}`;
	        const canSpawn = this.capabilities?.can_spawn_subprocesses !== false;

        const statusEl = document.getElementById('dashboard-openwebui-status');
        if (statusEl) {
            statusEl.textContent = statusText;
            statusEl.className = cls;
        }

	        const startBtn = document.getElementById('btn-quickstart-openwebui-start');
	        const stopBtn = document.getElementById('btn-quickstart-openwebui-stop');
	        if (startBtn) startBtn.disabled = !canSpawn || isRunning;
	        if (stopBtn) stopBtn.disabled = !canSpawn || !isRunning;

	        const stopBtnSystem = document.getElementById('btn-openwebui-stop');
	        if (stopBtnSystem) stopBtnSystem.disabled = !canSpawn || !isRunning;
	    },

    findOutputContainer(taskId) {
        if (taskId === this.currentBenchTaskId) return 'bench-output';
        if (taskId === this.currentBenchSweepTaskId) return 'bench-sweep-output';
        if (taskId === this.currentMemoryTaskId) return 'memory-output';
        if (taskId === this.currentMemorySweepTaskId) return 'memory-sweep-output';
        if (taskId === this.currentEvalHarnessTaskId) return 'eval-output';
        if (taskId === this.currentEvalCustomTaskId) return 'eval-output';
        if (taskId === this.currentWatcherTaskId) return 'watcher-output';
        return null;
    },

    checkAndResetTaskState(taskId) {
        if (taskId === this.currentBenchTaskId) {
            this.setTaskRunningState('bench', false);
            this.currentBenchTaskId = null;
        } else if (taskId === this.currentBenchSweepTaskId) {
            this.setTaskRunningState('bench-sweep', false);
            this.currentBenchSweepTaskId = null;
        } else if (taskId === this.currentMemoryTaskId) {
            this.setTaskRunningState('memory', false);
            this.updateSectionProgressUI('memory', 0, 'Idle');
            this.currentMemoryTaskId = null;
        } else if (taskId === this.currentMemorySweepTaskId) {
            this.setTaskRunningState('memory-sweep', false);
            this.updateSectionProgressUI('memory-sweep', 0, 'Idle');
            this.currentMemorySweepTaskId = null;
        } else if (taskId === this.currentEvalHarnessTaskId) {
            this.setTaskRunningState('eval-harness', false);
            this.updateSectionProgressUI('eval-harness', 0, 'Idle');
            this.currentEvalHarnessTaskId = null;
        } else if (taskId === this.currentEvalCustomTaskId) {
            this.setTaskRunningState('eval-custom', false);
            this.updateSectionProgressUI('eval-custom', 0, 'Idle');
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
            stopBtn.disabled = false;
            startBtn.disabled = false;
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
	        await this.loadLinksAndCapabilities();
	        await this.loadOverrides();
	        await this.loadSystemInfo();
	    },

	    async loadLinksAndCapabilities() {
	        try {
	            const links = await API.get('/api/system/links');
	            this.links = links;
	            this.capabilities = links;
	            this.applyLinks(links);
	            this.applyCapabilities(links);
	        } catch (e) {
	            console.error('Failed to load links:', e);
	        }
	    },

	    applyLinks(links) {
	        const swapApi = String(links.swap_api_url || '');
	        const swapUi = String(links.swap_ui_url || '');
	        const openWebUi = String(links.open_webui_url || '');

	        const setText = (id, value) => {
	            const el = document.getElementById(id);
	            if (el) el.textContent = value || '--';
	        };
	        const setHref = (id, href) => {
	            const el = document.getElementById(id);
	            if (!el) return;
	            if (!href) {
	                el.setAttribute('href', '#');
	                el.classList.add('disabled');
	            } else {
	                el.setAttribute('href', href);
	                el.classList.remove('disabled');
	            }
	        };

	        setText('dashboard-swap-api-url', swapApi);
	        setHref('dashboard-swap-ui-link', swapUi);
	        setText('dashboard-openwebui-url', openWebUi);
	        setHref('dashboard-openwebui-link', openWebUi);
	        setText('watcher-swap-api-url', swapApi);
	        setHref('system-openwebui-link', openWebUi);
	    },

	    applyCapabilities(caps) {
	        const canWriteConfigs = !!caps.can_write_configs;
	        const canWriteModels = !!caps.can_write_models;
	        const canSpawn = !!caps.can_spawn_subprocesses;

	        // Config section
	        const editor = document.getElementById('config-editor');
	        if (editor) editor.readOnly = !canWriteConfigs;
	        const saveBtn = document.getElementById('btn-save-config');
	        if (saveBtn) saveBtn.disabled = !canWriteConfigs || saveBtn.disabled;
	        const studioSaveBtn = document.getElementById('btn-config-save');
	        if (studioSaveBtn) studioSaveBtn.disabled = !canWriteConfigs || studioSaveBtn.disabled;
	        const studioTargetOverrideBtn = document.getElementById('cfg-target-override');
	        const studioTargetBaseBtn = document.getElementById('cfg-target-base');
	        if (studioTargetOverrideBtn) studioTargetOverrideBtn.disabled = !canWriteConfigs;
	        if (studioTargetBaseBtn) studioTargetBaseBtn.disabled = !canWriteConfigs;

	        const overrideNewBtn = document.getElementById('btn-new-override');
	        if (overrideNewBtn) overrideNewBtn.disabled = !canWriteConfigs;
	        const overrideHdrNewBtn = document.getElementById('btn-override-new');
	        if (overrideHdrNewBtn) overrideHdrNewBtn.disabled = !canWriteConfigs;
	        const overrideHdrEditBtn = document.getElementById('btn-override-edit');
	        if (overrideHdrEditBtn) overrideHdrEditBtn.disabled = !canWriteConfigs || !this.currentOverride;
	        const overrideHdrDupBtn = document.getElementById('btn-override-duplicate');
	        if (overrideHdrDupBtn) overrideHdrDupBtn.disabled = !canWriteConfigs || !this.currentOverride;

	        // Models section
	        const modelAddBtn = document.getElementById('btn-model-add');
	        if (modelAddBtn) modelAddBtn.disabled = !canWriteModels;
	        const modelUploadBtn = document.getElementById('btn-model-upload');
	        if (modelUploadBtn) modelUploadBtn.disabled = !canWriteModels;
	        const modelDownloadBtn = document.getElementById('btn-model-download');
	        if (modelDownloadBtn) modelDownloadBtn.disabled = !canSpawn;

	        // Endpoint / system subprocess features
	        const disableBtn = (id, disabled) => {
	            const el = document.getElementById(id);
	            if (el) el.disabled = !!disabled;
	        };

	        disableBtn('btn-quickstart-endpoint-start', !canSpawn);
	        disableBtn('btn-quickstart-endpoint-stop', !canSpawn);
	        disableBtn('btn-watcher-start', !canSpawn);
	        disableBtn('btn-watcher-stop', !canSpawn);
	        disableBtn('btn-quickstart-openwebui-start', !canSpawn);
	        disableBtn('btn-quickstart-openwebui-stop', !canSpawn);
	        disableBtn('btn-openwebui-stop', !canSpawn);

	        // Forms that trigger subprocesses
	        const disableForm = (formId) => {
	            const form = document.getElementById(formId);
	            if (!form) return;
	            form.querySelectorAll('input, select, button, textarea').forEach((el) => {
	                if (el.id === 'btn-copy-effective-config') return;
	                el.disabled = !canSpawn;
	            });
	        };
	        disableForm('bench-form');
	        disableForm('bench-sweep-form');
	        disableForm('memory-form');
	        disableForm('memory-sweep-form');
	        disableForm('eval-harness-form');
	        disableForm('eval-custom-form');
	        disableForm('openwebui-form');
	        disableForm('update-form');
	        disableForm('download-form');

	        if (String(caps.mode || '').toLowerCase() === 'gitops') {
	            Toast.info('GitOps mode: read-only UI (endpoint-only)');
	        }
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
	        const canWrite = this.capabilities?.can_write_configs !== false;
	        const editBtn = document.getElementById('btn-override-edit');
	        const dupBtn = document.getElementById('btn-override-duplicate');
	        if (editBtn) editBtn.disabled = !canWrite || !hasOverride;
	        if (dupBtn) dupBtn.disabled = !canWrite || !hasOverride;

            const selector = document.getElementById('override-selector');
            if (selector) {
                selector.classList.toggle('override-missing', !hasOverride);
                selector.classList.toggle('override-set', hasOverride);
            }

            const indicator = document.getElementById('override-indicator');
            if (indicator) {
                indicator.classList.toggle('is-none', !hasOverride);
                indicator.classList.toggle('is-set', hasOverride);
                indicator.textContent = hasOverride ? this.currentOverride : 'No override';
            }
	    },

        async maybeWarnNoOverride(actionLabel) {
            if (this.currentOverride) return true;

            const dismissedKey = 'llama-suite.noOverrideWarning.dismissed';
            const seenKey = 'llama-suite.noOverrideWarning.seen';

            try {
                if (localStorage.getItem(dismissedKey) === '1') return true;
            } catch {
                // ignore
            }

            try {
                if (sessionStorage.getItem(seenKey) === '1') return true;
            } catch {
                // ignore
            }

            if (this._noOverrideWarningInFlight) return await this._noOverrideWarningInFlight;

            const promise = new Promise((resolve) => {
                let done = false;

                const resolveOnce = (value) => {
                    if (done) return;
                    done = true;
                    this._noOverrideWarningInFlight = null;
                    cleanup();
                    resolve(value);
                };

                const markSeen = () => {
                    try {
                        sessionStorage.setItem(seenKey, '1');
                    } catch {
                        // ignore
                    }
                };

                const onOk = () => {
                    markSeen();
                    Modal.hide();
                    resolveOnce(true);
                };

                const onDontRemind = () => {
                    markSeen();
                    try {
                        localStorage.setItem(dismissedKey, '1');
                    } catch {
                        // ignore
                    }
                    Modal.hide();
                    resolveOnce(true);
                };

                const onClose = () => {
                    // Treat closing as acknowledgement so we don't block running.
                    markSeen();
                    resolveOnce(true);
                };

                const onKeyDown = (e) => {
                    if (e.key === 'Escape') {
                        Modal.hide();
                        onClose();
                    }
                };

                const overlayEl = document.getElementById('modal-overlay');
                const closeEl = document.getElementById('modal-close');

                const onOverlayClick = (e) => {
                    if (e.target === overlayEl) onClose();
                };

                const cleanup = () => {
                    document.removeEventListener('keydown', onKeyDown);
                    overlayEl?.removeEventListener('click', onOverlayClick);
                    closeEl?.removeEventListener('click', onClose);
                    document.getElementById('btn-no-override-ok')?.removeEventListener('click', onOk);
                    document.getElementById('btn-no-override-dont')?.removeEventListener('click', onDontRemind);
                };

                Modal.show(
                    'No override selected',
                    `
                        <div class="notice notice-warning">
                            <strong>${actionLabel || 'This action'}</strong> will run using <strong>base config only</strong>.
                            <div class="muted" style="margin-top: 6px;">
                                Overrides are recommended for machine-specific settings (GPU/VRAM, backend, model paths).
                                You can pick one in the header (top-right).
                            </div>
                        </div>
                    `,
                    `
                        <button class="btn btn-secondary" id="btn-no-override-dont">Don't remind again</button>
                        <button class="btn btn-primary" id="btn-no-override-ok">Okay</button>
                    `
                );

                document.getElementById('btn-no-override-ok')?.addEventListener('click', onOk);
                document.getElementById('btn-no-override-dont')?.addEventListener('click', onDontRemind);
                overlayEl?.addEventListener('click', onOverlayClick);
                closeEl?.addEventListener('click', onClose);
                document.addEventListener('keydown', onKeyDown);
            });

            this._noOverrideWarningInFlight = promise;
            return await promise;
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
             if (!window.ConfigStudio || typeof window.ConfigStudio.load !== 'function') {
                 throw new Error('ConfigStudio not available');
             }
             await window.ConfigStudio.load(this.currentOverride || '');
         } catch (e) {
             console.error('Failed to load Config Studio:', e);
             Toast.error(`Failed to load Config Studio: ${e.message}`);
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
            tbody.innerHTML = '<tr><td colspan="14" class="empty-message">No results found</td></tr>';
            return;
        }

        tbody.innerHTML = results.map((r, i) => `
            <tr class="${state.selected.has(r) ? 'selected' : ''}">
                <td class="col-checkbox"><input type="checkbox" onchange="App.handleResultsSelect('bench', ${state.data.indexOf(r)}, this.checked)" ${state.selected.has(r) ? 'checked' : ''}></td>
                <td title="${r.ModelName}">${r.ModelName}</td>
                <td>${r.ParameterSize || '-'}</td>
                <td>${r.Quantization || '-'}</td>
                <td>${(r.ContextSize ?? r._context_size ?? '-') || '-'}</td>
                <td>${(r.GpuLayers ?? r._gpu_layers ?? '-') || '-'}</td>
                <td>${r._kv_cache || `${r.CacheTypeK || '-'}\/${r.CacheTypeV || '-'}`}</td>
                <td>${(r.NCpuMoe ?? r._n_cpu_moe ?? '-') || '-'}</td>
                <td>${r._tokens_per_second ? r._tokens_per_second.toFixed(2) : '-'}</td>
                <td>${r._gpu_memory_gb ? r._gpu_memory_gb.toFixed(2) + ' GB' : '-'}</td>
                <td>${r._cpu_memory_gb ? r._cpu_memory_gb.toFixed(2) + ' GB' : '-'}</td>
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
            tbody.innerHTML = '<tr><td colspan="13" class="empty-message">No results found</td></tr>';
            return;
        }

        tbody.innerHTML = results.map((r, i) => `
            <tr class="${state.selected.has(r) ? 'selected' : ''}">
                <td class="col-checkbox"><input type="checkbox" onchange="App.handleResultsSelect('memory', ${state.data.indexOf(r)}, this.checked)" ${state.selected.has(r) ? 'checked' : ''}></td>
                <td title="${r.ModelName}">${r.ModelName}</td>
                <td>${r.ParameterSize || '-'}</td>
                <td>${r.Quantization || '-'}</td>
                <td>${(r.ContextSize ?? r._context_size ?? '-') || '-'}</td>
                <td>${(r.GpuLayers ?? r._gpu_layers ?? '-') || '-'}</td>
                <td>${r._kv_cache || `${r.CacheTypeK || '-'}\/${r.CacheTypeV || '-'}`}</td>
                <td>${(r.NCpuMoe ?? r._n_cpu_moe ?? '-') || '-'}</td>
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
            if (['_tokens_per_second', '_gpu_memory_gb', '_cpu_memory_gb', '_duration_seconds', '_context_size', '_gpu_layers', '_n_cpu_moe', 'gen_judge_score', 'count_total'].includes(col)) {
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
                            <tr><td><strong>CPU Memory</strong></td>${selected.map(r => `<td>${r._cpu_memory_gb?.toFixed(2) || '-'} GB</td>`).join('')}</tr>
                            <tr><td><strong>Duration</strong></td>${selected.map(r => `<td>${r._duration_seconds?.toFixed(2) || '-'} s</td>`).join('')}</tr>
                            <tr><td><strong>Context</strong></td>${selected.map(r => `<td>${r.ContextSize || '-'}</td>`).join('')}</tr>
                            <tr><td><strong>n-gpu</strong></td>${selected.map(r => `<td>${r.GpuLayers || '-'}</td>`).join('')}</tr>
                            <tr><td><strong>KV cache</strong></td>${selected.map(r => `<td>${r._kv_cache || `${r.CacheTypeK || '-'}\/${r.CacheTypeV || '-'}`}</td>`).join('')}</tr>
                            <tr><td><strong>n-cpu-moe</strong></td>${selected.map(r => `<td>${r.NCpuMoe || '-'}</td>`).join('')}</tr>
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
                            <tr><td><strong>Context</strong></td>${selected.map(r => `<td>${r.ContextSize || '-'}</td>`).join('')}</tr>
                            <tr><td><strong>n-gpu</strong></td>${selected.map(r => `<td>${r.GpuLayers || '-'}</td>`).join('')}</tr>
                            <tr><td><strong>KV cache</strong></td>${selected.map(r => `<td>${r._kv_cache || `${r.CacheTypeK || '-'}\/${r.CacheTypeV || '-'}`}</td>`).join('')}</tr>
                            <tr><td><strong>n-cpu-moe</strong></td>${selected.map(r => `<td>${r.NCpuMoe || '-'}</td>`).join('')}</tr>
                            <tr><td><strong>Quantization</strong></td>${selected.map(r => `<td>${r.Quantization || '-'}</td>`).join('')}</tr>
                            <tr><td><strong>Params</strong></td>${selected.map(r => `<td>${r.ParameterSize || '-'}</td>`).join('')}</tr>
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

        Modal.show(
            'Compare Results',
            content,
            '<button class="btn btn-secondary" onclick="Modal.hide()">Close</button>',
            { size: 'wide' }
        );
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

    async editCurrentOverride() {
        if (!this.currentOverride) {
            Toast.info('Select an override first');
            return;
        }

        // Prefer Config Studio (GUI) over raw YAML editing.
        if (this.currentSection !== 'config') {
            window.location.hash = 'config';
            return;
        }

        await this.loadConfig();
        try {
            if (window.ConfigStudio) {
                window.ConfigStudio.state.view = 'models';
                window.ConfigStudio.renderNav?.();
                window.ConfigStudio.render?.();
            }
        } catch {
            // ignore
        }
    },

    async duplicateCurrentOverride() {
        if (!this.currentOverride) {
            Toast.info('Select an override first');
            return;
        }
        const suggested = `${this.currentOverride}-copy`;
        this.showNewOverrideDialog({ name: suggested, copy_current: true });
    },

    showNewOverrideDialog(preset = {}) {
        const defaultName = preset.name || '';
        const canCopy = !!this.currentOverride;
        const defaultCopy = !!preset.copy_current;

        const copyRow = canCopy
            ? `
                <div class="form-group">
                    <label class="checkbox-label">
                        <input type="checkbox" id="new-override-copy-current" ${defaultCopy ? 'checked' : ''}>
                        Copy settings from current override (<code>${this.currentOverride}</code>)
                    </label>
                    <div class="muted" style="margin-top: 0.25rem;">
                        Creates a new override file starting from the current override YAML. Edit with Config Studio after creation.
                    </div>
                </div>
            `
            : `
                <div class="muted" style="margin-top: 0.25rem;">
                    No override selected; the new override will start empty (GUI-editable).
                </div>
            `;

        Modal.show('New Override', `
            <div class="form-grid" style="grid-template-columns: 1fr;">
                <div class="form-group">
                    <label for="new-override-name">Name</label>
                    <input type="text" id="new-override-name" placeholder="e.g., win-5090" value="">
                    <div class="muted" style="margin-top: 0.25rem;">Creates <code>configs/overrides/&lt;name&gt;.yaml</code></div>
                </div>
                ${copyRow}
            </div>
        `, `
            <button class="btn btn-secondary" onclick="Modal.hide()">Cancel</button>
            <button class="btn btn-primary" onclick="App.createOverrideFromDialog()">Create</button>
        `);

        const nameEl = document.getElementById('new-override-name');
        if (nameEl) nameEl.value = defaultName;
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
        const copyCurrent = !!document.getElementById('new-override-copy-current')?.checked;

        if (!name) {
            Toast.warning('Override name is required');
            return;
        }

        let content = this.buildOverrideTemplate();
        if (copyCurrent) {
            if (!this.currentOverride) {
                Toast.error('Select a current override to copy from');
                return;
            }
            const data = await API.get(`/api/config/overrides/${encodeURIComponent(this.currentOverride)}`);
            content = data.content || this.buildOverrideTemplate();
        }

        try {
            const created = await API.post('/api/config/overrides', { name, content });
            Toast.success(`Created override: ${created.name}`);
            Modal.hide();
            await this.loadOverrides();

            // Select the newly created override and persist it
            this.currentOverride = created.name;
            const select = document.getElementById('current-override');
            if (select) select.value = created.name;
            try {
                localStorage.setItem('llama-suite.override', created.name);
            } catch {
                // ignore
            }

            // Refresh the current section (and reload Config Studio if visible).
            await this.refreshCurrentSection();
        } catch (e) {
            Toast.error(`Failed to create override: ${e.message}`);
        }
    },

    async editOverride(name) {
        if (!name) return;

        // Prefer Config Studio (GUI) over raw YAML editing.
        if (name !== this.currentOverride) {
            this.currentOverride = name;
            const select = document.getElementById('current-override');
            if (select) select.value = name;
            try {
                localStorage.setItem('llama-suite.override', name);
            } catch {
                // ignore
            }
            this.updateOverrideActionsUI();
        }

        if (this.currentSection !== 'config') {
            window.location.hash = 'config';
            return;
        }
        await this.loadConfig();
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
            await this.maybeWarnNoOverride('Running benchmark');
            const selectedModel = document.getElementById('bench-model').value || '';
            if (!selectedModel) {
                const question = document.getElementById('bench-question').value;
                if (!confirm(`Run benchmark for ALL models?\n\nQuestion: ${question}`)) return;
            }
            const data = await API.post('/api/bench/run', {
                override: this.currentOverride || undefined,
                model: selectedModel || undefined,
                question: document.getElementById('bench-question').value,
                health_timeout: parseInt(document.getElementById('bench-timeout').value)
            });
            Toast.info(`Benchmark started: ${data.task_id}`);
            this.currentBenchTaskId = data.task_id;
            this.setTaskRunningState('bench', true);

            // Add to Task Manager
            this.taskManager.addTask(data.task_id, 'bench', 'Benchmark: ' + (selectedModel || 'All models'));

        } catch (e) {
            Toast.error(`Failed to start: ${e.message}`);
        }
    },

    async stopBenchmark() {
        const taskId = this.currentBenchTaskId;
        if (!taskId) return;

        const stopBtn = document.getElementById('btn-bench-stop');
        if (stopBtn) stopBtn.disabled = true;

        const ok = await this.cancelTask(taskId);
        if (!ok && stopBtn) stopBtn.disabled = false;
    },

    getSweepParamOptions() {
        try {
            const fields = window.ConfigStudio?.getSweepableFields?.() || [];
            if (Array.isArray(fields) && fields.length > 0) {
                return fields.map(f => ({
                    value: `${f.section}.${f.key}`,
                    label: `${f.section}.${f.key} (${f.label || f.key})`,
                    valueType: (f.type === 'int' ? 'int' : (f.type === 'float' ? 'float' : (f.type === 'bool' ? 'bool' : 'str'))),
                }));
            }
        } catch {
            // ignore
        }

        // Fallback if Config Studio isn't loaded yet.
        return [
            { value: 'cmd.ctx-size', label: 'cmd.ctx-size (Context size)', valueType: 'int' },
            { value: 'cmd.cache-type-k', label: 'cmd.cache-type-k', valueType: 'str' },
            { value: 'cmd.cache-type-v', label: 'cmd.cache-type-v', valueType: 'str' },
            { value: 'cmd.parallel', label: 'cmd.parallel', valueType: 'int' },
            { value: 'cmd.jinja', label: 'cmd.jinja', valueType: 'bool' },
            { value: 'sampling.temp', label: 'sampling.temp', valueType: 'float' },
            { value: 'sampling.top-p', label: 'sampling.top-p', valueType: 'float' },
        ];
    },

    setupSweepsUI() {
        if (this._sweepsUiBound) return;
        this._sweepsUiBound = true;

        const bind = (id, event, handler) => {
            const el = document.getElementById(id);
            if (el) el.addEventListener(event, handler);
        };

        bind('btn-bench-sweep-add-dim', 'click', () => this.addSweepDimRow('bench'));
        bind('btn-memory-sweep-add-dim', 'click', () => this.addSweepDimRow('memory'));

        bind('bench-sweep-scope', 'change', () => this.updateSweepRunCount('bench'));
        bind('bench-sweep-filter', 'input', () => this.updateSweepRunCount('bench'));
        bind('memory-sweep-scope', 'change', () => this.updateSweepRunCount('memory'));
        bind('memory-sweep-filter', 'input', () => this.updateSweepRunCount('memory'));

        // Seed one dimension row by default.
        if (document.getElementById('bench-sweep-dims')?.children.length === 0) this.addSweepDimRow('bench');
        if (document.getElementById('memory-sweep-dims')?.children.length === 0) this.addSweepDimRow('memory');
    },

    addSweepDimRow(kind) {
        const containerId = kind === 'bench' ? 'bench-sweep-dims' : 'memory-sweep-dims';
        const container = document.getElementById(containerId);
        if (!container) return;

        const row = document.createElement('div');
        row.className = 'sweep-dim-row';

        const select = document.createElement('select');
        select.className = 'sweep-param';
        for (const opt of this.getSweepParamOptions()) {
            const o = document.createElement('option');
            o.value = opt.value;
            o.textContent = opt.label;
            o.dataset.valueType = opt.valueType;
            select.appendChild(o);
        }

        const values = document.createElement('input');
        values.type = 'text';
        values.className = 'sweep-values';
        values.placeholder = 'e.g. 8192,16384 or 8192..32768..8192';

        const count = document.createElement('div');
        count.className = 'muted sweep-count';
        count.textContent = '0';

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'btn btn-secondary btn-sm';
        removeBtn.textContent = 'Remove';

        row.appendChild(select);
        row.appendChild(values);
        row.appendChild(count);
        row.appendChild(removeBtn);

        const update = () => {
            const parsed = this.parseSweepValues(select.selectedOptions?.[0]?.dataset?.valueType || 'str', values.value);
            count.textContent = parsed.ok ? String(parsed.count) : '!';
            this.updateSweepRunCount(kind);
        };
        select.addEventListener('change', update);
        values.addEventListener('input', update);
        removeBtn.addEventListener('click', () => {
            row.remove();
            this.updateSweepRunCount(kind);
        });

        container.appendChild(row);
        update();
    },

    parseSweepValues(valueType, raw) {
        const text = String(raw || '').trim();
        if (!text) return { ok: false, count: 0, values: [], range: null, error: 'empty' };

        // range syntax: start..end..step (inclusive)
        if (text.includes('..')) {
            const parts = text.split('..').map(s => s.trim()).filter(Boolean);
            if (parts.length < 3) return { ok: false, count: 0, values: [], range: null, error: 'range expects start..end..step' };
            const start = Number(parts[0]);
            const end = Number(parts[1]);
            const step = Number(parts[2]);
            if (!Number.isFinite(start) || !Number.isFinite(end) || !Number.isFinite(step) || step <= 0) {
                return { ok: false, count: 0, values: [], range: null, error: 'invalid range numbers' };
            }
            return { ok: true, count: Math.max(0, Math.floor((end - start) / step) + 1), values: [], range: { start, end, step } };
        }

        const parts = text.split(',').map(s => s.trim()).filter(Boolean);
        const vals = [];
        for (const p of parts) {
            if (valueType === 'int') {
                const n = parseInt(p, 10);
                if (Number.isNaN(n)) return { ok: false, count: 0, values: [], range: null, error: 'invalid int' };
                vals.push(n);
            } else if (valueType === 'float') {
                const n = parseFloat(p);
                if (Number.isNaN(n)) return { ok: false, count: 0, values: [], range: null, error: 'invalid float' };
                vals.push(n);
            } else if (valueType === 'bool') {
                const v = p.toLowerCase();
                if (v === 'true' || v === '1' || v === 'yes' || v === 'y') vals.push(true);
                else if (v === 'false' || v === '0' || v === 'no' || v === 'n') vals.push(false);
                else return { ok: false, count: 0, values: [], range: null, error: 'invalid bool' };
            } else {
                vals.push(p);
            }
        }
        return { ok: true, count: vals.length, values: vals, range: null };
    },

    updateSweepRunCount(kind) {
        const scopeId = kind === 'bench' ? 'bench-sweep-scope' : 'memory-sweep-scope';
        const filterId = kind === 'bench' ? 'bench-sweep-filter' : 'memory-sweep-filter';
        const dimsId = kind === 'bench' ? 'bench-sweep-dims' : 'memory-sweep-dims';
        const labelId = kind === 'bench' ? 'bench-sweep-run-count' : 'memory-sweep-run-count';

        const label = document.getElementById(labelId);
        const dims = document.getElementById(dimsId);
        if (!label || !dims) return;

        const scope = document.getElementById(scopeId)?.value || 'ALL';
        const filter = String(document.getElementById(filterId)?.value || '').trim();

        let modelCount = 0;
        if (scope === 'SELECTED') {
            modelCount = window.ConfigStudio?.getSelectedModels?.()?.length || 0;
        } else {
            const all = window.ConfigStudio?.state?.meta?.models;
            modelCount = Array.isArray(all) ? all.length : 0;
        }

        let combos = 1;
        dims.querySelectorAll('.sweep-dim-row').forEach(row => {
            const sel = row.querySelector('.sweep-param');
            const inp = row.querySelector('.sweep-values');
            const valueType = sel?.selectedOptions?.[0]?.dataset?.valueType || 'str';
            const parsed = this.parseSweepValues(valueType, inp?.value || '');
            combos *= Math.max(0, parsed.count || 0);
        });

        const runs = modelCount && combos ? (modelCount * combos) : 0;
        const filterNote = filter ? ` (filter: ${filter})` : '';
        label.textContent = `${runs} runs${filterNote}`;
    },

    collectSweepDimensions(kind) {
        const dimsId = kind === 'bench' ? 'bench-sweep-dims' : 'memory-sweep-dims';
        const dims = document.getElementById(dimsId);
        if (!dims) return [];

        const out = [];
        dims.querySelectorAll('.sweep-dim-row').forEach(row => {
            const sel = row.querySelector('.sweep-param');
            const inp = row.querySelector('.sweep-values');
            const pathStr = sel?.value || '';
            const valueType = sel?.selectedOptions?.[0]?.dataset?.valueType || 'str';
            const parsed = this.parseSweepValues(valueType, inp?.value || '');
            if (!pathStr || !parsed.ok) return;
            const path = pathStr.split('.').filter(Boolean);
            const dim = { path, value_type: valueType, values: null, range: null };
            if (parsed.range) dim.range = parsed.range;
            else dim.values = parsed.values;
            out.push(dim);
        });
        return out;
    },

    async runBenchSweep() {
        const output = document.getElementById('bench-sweep-output');
        if (output) output.innerHTML = '';

        const table = document.getElementById('bench-sweep-results-table');
        if (table) {
            table.querySelector('thead').innerHTML = '';
            table.querySelector('tbody').innerHTML = '';
        }

        try {
            await this.maybeWarnNoOverride('Running benchmark sweep');

            const scope = document.getElementById('bench-sweep-scope')?.value || 'ALL';
            const filter = String(document.getElementById('bench-sweep-filter')?.value || '').trim();
            const dims = this.collectSweepDimensions('bench');
            if (dims.length === 0) {
                Toast.error('Add at least one sweep dimension');
                return;
            }

            const models = (scope === 'SELECTED')
                ? (window.ConfigStudio?.getSelectedModels?.() || [])
                : 'ALL';

            if (scope === 'SELECTED' && (!Array.isArray(models) || models.length === 0)) {
                Toast.error('No selected models (select in Config Studio - Models)');
                return;
            }

            const question = document.getElementById('bench-sweep-question')?.value || 'What is the capital of France?';
            const healthTimeout = parseInt(document.getElementById('bench-sweep-timeout')?.value || '120', 10);

            const data = await API.post('/api/sweeps/run', {
                task_type: 'bench',
                baseline_override: this.currentOverride || undefined,
                models,
                filter_string: filter || undefined,
                dimensions: dims,
                health_timeout: healthTimeout,
                question,
            });

            Toast.info(`Sweep started: ${data.task_id}`);
            this.currentBenchSweepTaskId = data.task_id;
            this.setTaskRunningState('bench-sweep', true);
            this.registerTaskContainer(data.task_id, 'bench-sweep-output');
            this.taskManager.addTask(data.task_id, 'bench', 'Benchmark Sweep');
        } catch (e) {
            Toast.error(`Failed to start sweep: ${e.message}`);
        }
    },

    async stopBenchSweep() {
        const taskId = this.currentBenchSweepTaskId;
        if (!taskId) return;
        const stopBtn = document.getElementById('btn-bench-sweep-stop');
        if (stopBtn) stopBtn.disabled = true;
        const ok = await this.cancelTask(taskId);
        if (!ok && stopBtn) stopBtn.disabled = false;
    },

    async runMemoryScan() {
        const output = document.getElementById('memory-output');
        output.innerHTML = '';

        try {
            await this.maybeWarnNoOverride('Running memory scan');
            const selectedModel = document.getElementById('memory-model').value || '';
            if (!selectedModel) {
                if (!confirm('Run memory scan for ALL models? This can take a while.')) return;
            }

            this.updateSectionProgressUI('memory', -1, 'Starting...');
            const data = await API.post('/api/memory/run', {
                override: this.currentOverride || undefined,
                model: selectedModel || undefined,
                health_timeout: parseInt(document.getElementById('memory-timeout').value)
            });
            Toast.info(`Memory scan started: ${data.task_id}`);
            this.currentMemoryTaskId = data.task_id;
            this.setTaskRunningState('memory', true);

            // Add to Task Manager
            this.taskManager.addTask(data.task_id, 'memory', 'Memory Scan: ' + (selectedModel || 'All models'));

        } catch (e) {
            this.updateSectionProgressUI('memory', 0, 'Idle');
            Toast.error(`Failed to start: ${e.message}`);
        }
    },

    async stopMemoryScan() {
        const taskId = this.currentMemoryTaskId;
        if (!taskId) return;

        const stopBtn = document.getElementById('btn-memory-stop');
        if (stopBtn) stopBtn.disabled = true;

        this.updateSectionProgressUI('memory', -1, 'Cancelling...');
        const ok = await this.cancelTask(taskId);
        if (!ok && stopBtn) stopBtn.disabled = false;
    },

    async runMemorySweep() {
        const output = document.getElementById('memory-sweep-output');
        if (output) output.innerHTML = '';

        const table = document.getElementById('memory-sweep-results-table');
        if (table) {
            table.querySelector('thead').innerHTML = '';
            table.querySelector('tbody').innerHTML = '';
        }

        try {
            await this.maybeWarnNoOverride('Running memory sweep');

            const scope = document.getElementById('memory-sweep-scope')?.value || 'ALL';
            const filter = String(document.getElementById('memory-sweep-filter')?.value || '').trim();
            const dims = this.collectSweepDimensions('memory');
            if (dims.length === 0) {
                Toast.error('Add at least one sweep dimension');
                return;
            }

            const models = (scope === 'SELECTED')
                ? (window.ConfigStudio?.getSelectedModels?.() || [])
                : 'ALL';

            if (scope === 'SELECTED' && (!Array.isArray(models) || models.length === 0)) {
                Toast.error('No selected models (select in Config Studio - Models)');
                return;
            }

            const healthTimeout = parseInt(document.getElementById('memory-sweep-timeout')?.value || '120', 10);
            this.updateSectionProgressUI('memory-sweep', -1, 'Starting...');

            const data = await API.post('/api/sweeps/run', {
                task_type: 'memory',
                baseline_override: this.currentOverride || undefined,
                models,
                filter_string: filter || undefined,
                dimensions: dims,
                health_timeout: healthTimeout,
            });

            Toast.info(`Sweep started: ${data.task_id}`);
            this.currentMemorySweepTaskId = data.task_id;
            this.setTaskRunningState('memory-sweep', true);
            this.registerTaskContainer(data.task_id, 'memory-sweep-output');
            this.taskManager.addTask(data.task_id, 'memory', 'Memory Sweep');
        } catch (e) {
            this.updateSectionProgressUI('memory-sweep', 0, 'Idle');
            Toast.error(`Failed to start sweep: ${e.message}`);
        }
    },

    async stopMemorySweep() {
        const taskId = this.currentMemorySweepTaskId;
        if (!taskId) return;
        const stopBtn = document.getElementById('btn-memory-sweep-stop');
        if (stopBtn) stopBtn.disabled = true;
        this.updateSectionProgressUI('memory-sweep', -1, 'Cancelling...');
        const ok = await this.cancelTask(taskId);
        if (!ok && stopBtn) stopBtn.disabled = false;
    },

    async loadSweepResults(taskId, tableId) {
        const table = document.getElementById(tableId);
        if (!table) return;

        try {
            const data = await API.get(`/api/sweeps/task/${encodeURIComponent(taskId)}/results?offset=0&limit=200`);
            const rows = data.rows || [];
            const thead = table.querySelector('thead');
            const tbody = table.querySelector('tbody');
            if (!thead || !tbody) return;

            if (rows.length === 0) {
                thead.innerHTML = '';
                tbody.innerHTML = '<tr><td class="muted">No results</td></tr>';
                return;
            }

            const paramKeys = new Set();
            const metricKeys = new Set();
            rows.forEach(r => {
                Object.keys(r).forEach(k => {
                    if (k.startsWith('param.')) paramKeys.add(k);
                    if (k.startsWith('metric.')) metricKeys.add(k);
                });
            });

            const preferredMetrics = [
                'metric.TokensPerSecond',
                'metric.BenchStatus',
                'metric.DurationSeconds',
                'metric.ScanStatus',
                'metric.GpuMemoryGB',
                'metric.CpuMemoryGB',
            ];
            const metricCols = preferredMetrics.filter(k => metricKeys.has(k));
            const otherMetricCols = Array.from(metricKeys).filter(k => !metricCols.includes(k)).sort().slice(0, 4);
            const cols = [
                'model',
                'variant_label',
                'returncode',
                ...Array.from(paramKeys).sort(),
                ...metricCols,
                ...otherMetricCols,
            ];

            const headerLabel = (k) => {
                if (k === 'model') return 'Model';
                if (k === 'variant_label') return 'Variant';
                if (k === 'returncode') return 'RC';
                if (k.startsWith('param.')) return k.slice('param.'.length);
                if (k.startsWith('metric.')) return k.slice('metric.'.length);
                return k;
            };

            thead.innerHTML = `<tr>${cols.map(c => `<th>${headerLabel(c)}</th>`).join('')}</tr>`;
            tbody.innerHTML = rows.map(r => `<tr>${cols.map(c => `<td>${(r[c] ?? '').toString()}</td>`).join('')}</tr>`).join('');
        } catch (e) {
            console.error('Failed to load sweep results:', e);
            Toast.error(`Failed to load sweep results: ${e.message}`);
        }
    },

    async runEvalHarness() {
        const output = document.getElementById('eval-output');
        output.innerHTML = '';

        try {
            await this.maybeWarnNoOverride('Running eval harness');
            const tasks = document.getElementById('eval-tasks').value;
            const selectedModel = document.getElementById('eval-model').value || '';
            if (!selectedModel) {
                if (!confirm(`Run evaluation harness for ALL models?\n\nTasks: ${tasks}`)) return;
            }

            this.updateSectionProgressUI('eval-harness', -1, 'Starting...');
            const data = await API.post('/api/eval/harness/run', {
                override: this.currentOverride || undefined,
                model: selectedModel || undefined,
                tasks: tasks,
                limit: document.getElementById('eval-limit').value ? parseFloat(document.getElementById('eval-limit').value) : undefined,
                num_fewshot: document.getElementById('eval-fewshot').value ? parseInt(document.getElementById('eval-fewshot').value) : undefined,
                batch_size: document.getElementById('eval-batch').value
            });
            Toast.info(`Eval-harness started: ${data.task_id}`);
            this.currentEvalHarnessTaskId = data.task_id;
            this.setTaskRunningState('eval-harness', true);

            // Add to Task Manager
            this.taskManager.addTask(data.task_id, 'eval', `Eval Harness (${selectedModel || 'All models'}): ${tasks}`);

        } catch (e) {
            this.updateSectionProgressUI('eval-harness', 0, 'Idle');
            Toast.error(`Failed to start: ${e.message}`);
        }
    },

    async stopEvalHarness() {
        const taskId = this.currentEvalHarnessTaskId;
        if (!taskId) return;

        const stopBtn = document.getElementById('btn-eval-harness-stop');
        if (stopBtn) stopBtn.disabled = true;

        this.updateSectionProgressUI('eval-harness', -1, 'Cancelling...');
        const ok = await this.cancelTask(taskId);
        if (!ok && stopBtn) stopBtn.disabled = false;
    },

    async runCustomEval() {
        const output = document.getElementById('eval-output');
        output.innerHTML = '';

        try {
            await this.maybeWarnNoOverride('Running custom eval');
            const dataset = document.getElementById('custom-dataset').value;
            const selectedModel = document.getElementById('custom-model').value || '';
            if (!selectedModel) {
                if (!confirm(`Run custom eval for ALL models?\n\nDataset: ${dataset}`)) return;
            }

            this.updateSectionProgressUI('eval-custom', -1, 'Starting...');
            const data = await API.post('/api/eval/custom/run', {
                override: this.currentOverride || undefined,
                model: selectedModel || undefined,
                dataset: dataset,
                max_tasks: document.getElementById('custom-max-tasks').value ? parseInt(document.getElementById('custom-max-tasks').value) : undefined,
                temperature: document.getElementById('custom-temp').value ? parseFloat(document.getElementById('custom-temp').value) : undefined
            });
            Toast.info(`Custom eval started: ${data.task_id}`);
            this.currentEvalCustomTaskId = data.task_id;
            this.setTaskRunningState('eval-custom', true);

            // Add to Task Manager
            this.taskManager.addTask(data.task_id, 'eval', `Custom Eval (${selectedModel || 'All models'}): ${dataset}`);

        } catch (e) {
            this.updateSectionProgressUI('eval-custom', 0, 'Idle');
            Toast.error(`Failed to start: ${e.message}`);
        }
    },

    async stopCustomEval() {
        const taskId = this.currentEvalCustomTaskId;
        if (!taskId) return;

        const stopBtn = document.getElementById('btn-eval-custom-stop');
        if (stopBtn) stopBtn.disabled = true;

        this.updateSectionProgressUI('eval-custom', -1, 'Cancelling...');
        const ok = await this.cancelTask(taskId);
        if (!ok && stopBtn) stopBtn.disabled = false;
    },

    async startWatcher() {
        const output = document.getElementById('watcher-output');
        output.innerHTML = '';

        try {
            await this.maybeWarnNoOverride('Starting endpoint');
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

            await this.maybeWarnNoOverride('Downloading models');
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
            return true;
        } catch (e) {
            Toast.error('Failed to cancel task');
            return false;
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
window.runBenchSweep = () => App.runBenchSweep();
window.stopBenchSweep = () => App.stopBenchSweep();
window.runMemorySweep = () => App.runMemorySweep();
window.stopMemorySweep = () => App.stopMemorySweep();
window.stopEvalHarness = () => App.stopEvalHarness();
window.stopCustomEval = () => App.stopCustomEval();
