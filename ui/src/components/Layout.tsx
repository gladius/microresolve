import { Link, NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useAppStore } from '@/store';
import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '@/api/client';

// 2026 layout: sidebar + fullscreen main area. No top navbar bloat.
// Namespace switcher lives in the sidebar header. Domains are managed
// inside NamespacesPage/DomainsPage, not global nav.

type NavItem = {
  to: string;
  label: string;
  icon: string;
  hint?: string;
  badge?: number;
  /// When true, badge is rendered even at 0 (muted styling). Use for
  /// "live status" indicators where 0 is informative ("yes the panel
  /// works, nothing's connected"). Default: hide-when-0 (review-style
  /// unread count where 0 means clear).
  showZero?: boolean;
  /// When set, the row renders a muted "off" or "partial" pill. Used to
  /// flag layers disabled per-namespace so it's visible from anywhere in
  /// the app, not just the namespace settings page.
  layerStatus?: 'off' | 'partial';
};

type ConnectedClient = {
  name: string;
  namespaces: string[];
  tick_interval_secs: number;
  library_version: string | null;
  last_seen_ms: number;
  age_ms: number;
  expires_in_ms: number;
};

export default function Layout() {
  const { settings, setSelectedNamespaceId, setSelectedDomain, layerStatus, setLayerStatus } = useAppStore();
  const navigate = useNavigate();

  const [namespaces,    setNamespaces]    = useState<string[]>(['default']);
  const [showNsMenu,    setShowNsMenu]    = useState(false);
  const [nsFilter,      setNsFilter]      = useState('');
  const [collapsed,     setCollapsed]     = useState(false);
  const [reviewPending, setReviewPending] = useState(0);
  const [showBackupMenu, setShowBackupMenu] = useState(false);
  const [restoreStatus,  setRestoreStatus]  = useState<string | null>(null);
  const [appVersion,     setAppVersion]     = useState<string | null>(null);
  const [connectedClients, setConnectedClients] = useState<ConnectedClient[]>([]);
  const nsMenuRef     = useRef<HTMLDivElement>(null);
  const backupMenuRef = useRef<HTMLDivElement>(null);

  // Poll the connected-clients roster every 5s. Lazy-GC happens server-side
  // on each read, so the count is always fresh — no client-side cleanup.
  useEffect(() => {
    let alive = true;
    const fetchClients = async () => {
      try {
        const r = await fetch('/api/connected_clients');
        if (!r.ok) return;
        const d = await r.json();
        if (alive) setConnectedClients(d.clients ?? []);
      } catch { /* noop */ }
    };
    fetchClients();
    const t = setInterval(fetchClients, 5000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  useEffect(() => {
    api.listNamespaces().then(ns => {
      setNamespaces(ns.map(n => n.id));
      const active = ns.find(n => n.id === settings.selectedNamespaceId);
      if (active) {
        setLayerStatus({
          l0:  active.l0_enabled       ?? true,
          l1m: active.l1_morphology    ?? true,
          l1s: active.l1_synonym       ?? true,
          l1a: active.l1_abbreviation  ?? true,
        });
      }
    }).catch(() => {});
    fetch('/api/version').then(r => r.json()).then(d => setAppVersion(d.app_version)).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.selectedNamespaceId]);

  // Poll review queue count + refresh on SSE events
  const refreshReviewCount = useCallback(() => {
    api.getReviewStats().then(s => setReviewPending(s.pending)).catch(() => {});
  }, []);

  useEffect(() => {
    refreshReviewCount();
    const es = new EventSource('/api/events');
    es.onmessage = (e) => {
      try {
        const ev = JSON.parse(e.data);
        if (ev.type === 'item_queued' || ev.type === 'fix_applied' || ev.type === 'escalated') {
          setTimeout(refreshReviewCount, 600);
        }
      } catch { /* */ }
    };
    return () => es.close();
  }, [refreshReviewCount]);

  useEffect(() => {
    if (!showNsMenu) return;
    const handler = (e: MouseEvent) => {
      if (nsMenuRef.current && !nsMenuRef.current.contains(e.target as Node)) {
        setShowNsMenu(false);
        setNsFilter('');
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [showNsMenu]);

  useEffect(() => {
    if (!showBackupMenu) return;
    const handler = (e: MouseEvent) => {
      if (backupMenuRef.current && !backupMenuRef.current.contains(e.target as Node)) {
        setShowBackupMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [showBackupMenu]);

  const handleDownloadBackup = () => {
    setShowBackupMenu(false);
    const a = document.createElement('a');
    a.href = '/api/state/export';
    a.download = '';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleRestoreBackup = () => {
    setShowBackupMenu(false);
    if (!window.confirm('This will replace ALL current namespaces. Continue?')) return;
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.zip';
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      const form = new FormData();
      form.append('file', file);
      setRestoreStatus('Restoring…');
      try {
        const res = await fetch('/api/state/import', { method: 'POST', body: form });
        if (!res.ok) {
          const msg = await res.text();
          setRestoreStatus(null);
          alert(`Restore failed: ${msg}`);
          return;
        }
        setRestoreStatus(null);
        window.location.reload();
      } catch (err) {
        setRestoreStatus(null);
        alert(`Restore failed: ${err}`);
      }
    };
    input.click();
  };

  const switchNamespace = (namespaceId: string) => {
    setSelectedNamespaceId(namespaceId);
    setSelectedDomain('');
    setShowNsMenu(false);
    window.location.reload();
  };

  const activeNs = settings.selectedNamespaceId;

  type NavGroup = { label: string; items: NavItem[] };
  const NAV_GROUPS: NavGroup[] = [
    {
      label: 'Live',
      items: [
        { to: '/connected', label: 'Connected', icon: '⚡', hint: 'Library clients currently syncing', badge: connectedClients.length, showZero: true },
      ],
    },
    {
      label: 'Build',
      items: [
        { to: '/l2', label: 'L2 — Intents', icon: '◆',
          hint: 'Manage intents, training phrases, metadata' },
        { to: '/l1', label: 'L1 — Lexical', icon: '⧉',
          hint: 'Morphology, synonym, and abbreviation edges',
          layerStatus: (!layerStatus.l1m && !layerStatus.l1s && !layerStatus.l1a)
            ? 'off'
            : (layerStatus.l1m && layerStatus.l1s && layerStatus.l1a) ? undefined : 'partial' },
        { to: '/l0', label: 'L0 — Spelling', icon: '∼',
          hint: 'Typo correction — vocabulary inspector + live tester',
          layerStatus: layerStatus.l0 ? undefined : 'off' },
      ],
    },
    {
      label: 'Train',
      items: [
        { to: '/import',   label: 'Import',   icon: '↓', hint: 'Import from OpenAPI, MCP, and more' },
        { to: '/simulate', label: 'Simulate', icon: '◎', hint: 'LLM generates queries, system learns from failures' },
        { to: '/review',   label: 'Review',   icon: '✦', hint: 'Triage flagged queries from production', badge: reviewPending || undefined },
      ],
    },
    {
      label: 'Test',
      items: [
        { to: '/resolve', label: 'Resolve', icon: '▸', hint: 'Probe queries through the full L0→L1→L2→L3 pipeline' },
      ],
    },
    {
      label: 'Manage',
      items: [
        { to: '/namespaces', label: 'Namespaces', icon: '▦', hint: 'Create, switch, and delete namespaces' },
        { to: '/history',    label: 'Git History', icon: '⏱', hint: 'Browse commits, diff between revisions, roll back the namespace to any prior state' },
        { to: '/models',     label: 'Models',     icon: '⬡', hint: 'Application-wide routing model registry' },
        { to: '/languages',  label: 'Languages',  icon: '◌', hint: 'Application-wide phrase-generation languages' },
        { to: '/auth',       label: 'Auth Keys',  icon: '⚿', hint: 'API keys for connected libraries' },
        { to: '/settings',   label: 'Settings',   icon: '⚙', hint: 'LLM config and data management' },
      ],
    },
  ];

  const sidebarWidth = collapsed ? 'w-14' : 'w-56';

  return (
    <div className="h-screen flex overflow-hidden bg-zinc-950">
      {/* Sidebar */}
      <aside className={`${sidebarWidth} shrink-0 flex flex-col border-r border-zinc-800 bg-zinc-950 transition-all duration-150`}>
        {/* Brand + collapse */}
        <div className="h-12 flex items-center px-3 border-b border-zinc-800">
          <Link to="/" className="hover:opacity-80 transition-opacity" aria-label="Home" title="Home">
            {collapsed ? (
              <span className="text-emerald-400 font-bold text-xl leading-none">μ</span>
            ) : (
              <span className="flex items-baseline gap-1.5">
                <span className="flex items-baseline gap-1">
                  <span className="text-emerald-400 font-bold text-lg leading-none">μ</span>
                  <span className="text-zinc-100 font-bold text-base tracking-tight">Resolve</span>
                </span>
                <span className="text-[9px] font-semibold text-zinc-500 uppercase tracking-[0.15em] leading-none pb-px">Studio</span>
              </span>
            )}
          </Link>
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="ml-auto text-zinc-600 hover:text-zinc-300 text-xs px-2 py-1 rounded hover:bg-zinc-800 transition-colors"
            title={collapsed ? 'Expand' : 'Collapse'}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? '›' : '‹'}
          </button>
        </div>

        {/* Namespace switcher */}
        <div ref={nsMenuRef} className="px-2 py-2 border-b border-zinc-800/60 relative">
          <button
            onClick={() => setShowNsMenu(!showNsMenu)}
            className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs text-zinc-300 hover:bg-zinc-800/70 border border-zinc-800 transition-colors"
            title={`Namespace: ${activeNs}`}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 shrink-0" />
            {!collapsed && (
              <>
                <span className="font-mono text-blue-300 truncate">{activeNs}</span>
                <svg className="w-3 h-3 ml-auto shrink-0 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </>
            )}
          </button>

          {showNsMenu && (
            <div className={`absolute ${collapsed ? 'left-full ml-1' : 'left-2 right-2'} top-full mt-1 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl z-50 min-w-[12rem] flex flex-col max-h-72`}>
              <div className="px-2 pt-2 pb-1 shrink-0">
                <input
                  autoFocus
                  value={nsFilter}
                  onChange={e => setNsFilter(e.target.value)}
                  placeholder="Filter namespaces..."
                  className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div className="overflow-y-auto flex-1 py-1">
                {namespaces.filter(ns => !nsFilter || ns.includes(nsFilter)).map(ns => (
                  <button
                    key={ns}
                    onClick={() => { setNsFilter(''); switchNamespace(ns); }}
                    className={`w-full text-left px-3 py-1.5 text-sm hover:bg-zinc-800 transition-colors ${
                      ns === activeNs ? 'text-blue-400 font-medium' : 'text-zinc-300'
                    }`}
                  >
                    {ns}
                  </button>
                ))}
                {namespaces.filter(ns => !nsFilter || ns.includes(nsFilter)).length === 0 && (
                  <div className="px-3 py-2 text-xs text-zinc-600">No match</div>
                )}
              </div>
              <div className="border-t border-zinc-700 shrink-0 pt-1 px-2 pb-1">
                <button
                  onClick={() => { setShowNsMenu(false); navigate('/namespaces'); }}
                  className="w-full text-left text-xs text-zinc-400 hover:text-emerald-400 px-1 py-1"
                >
                  + Manage namespaces
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Nav groups */}
        <nav className="flex-1 py-2 overflow-y-auto">
          {NAV_GROUPS.map((group, gi) => (
            <div key={group.label}>
              {/* Group divider — only between groups, not before first */}
              {gi > 0 && <div className="mx-2 my-1 border-t border-zinc-800/60" />}
              {/* Group label — hidden when collapsed */}
              {!collapsed && (
                <div className="px-4 pt-2 pb-1 text-[9px] font-semibold uppercase tracking-widest text-zinc-600">
                  {group.label}
                </div>
              )}
              {group.items.map(item => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === '/resolve'}
                  className={({ isActive }) =>
                    `relative mx-2 my-0.5 px-2 py-1.5 rounded flex items-center gap-2.5 text-sm transition-colors ${
                      isActive
                        ? 'bg-zinc-800 text-zinc-100'
                        : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
                    }`
                  }
                  title={collapsed ? item.label : item.hint}
                >
                  <span className={`w-4 text-center shrink-0 ${
                    item.showZero && item.badge && item.badge > 0
                      ? 'text-amber-400'
                      : 'text-zinc-500'
                  }`}>{item.icon}</span>
                  {!collapsed && <span className="truncate flex-1">{item.label}</span>}
                  {!collapsed && item.layerStatus && (
                    <span
                      className={`text-[8px] px-1 py-px rounded font-bold uppercase tracking-wider flex-shrink-0 ${
                        item.layerStatus === 'off'
                          ? 'bg-zinc-800 text-zinc-500'
                          : 'bg-amber-500/15 text-amber-400/90'
                      }`}
                      title={item.layerStatus === 'off' ? 'Disabled for this namespace' : 'Partially disabled for this namespace'}
                    >
                      {item.layerStatus === 'off' ? 'off' : 'partial'}
                    </span>
                  )}
                  {!collapsed && (item.badge !== undefined && (item.badge > 0 || item.showZero)) && (
                    <span className={`text-[9px] px-1.5 py-0.5 rounded-full font-bold flex-shrink-0 ${
                      item.badge && item.badge > 0
                        ? (item.showZero
                            ? 'bg-emerald-500/20 text-emerald-300'   // live indicator, active
                            : 'bg-amber-500/20 text-amber-400')      // unread/pending count
                        : 'bg-zinc-800 text-zinc-500'                // showZero=true, count is 0 → muted
                    }`}>
                      {item.badge}
                    </span>
                  )}
                  {collapsed && (item.badge !== undefined && item.badge > 0) && (
                    <span className={`absolute top-0.5 right-0.5 w-2 h-2 rounded-full ${
                      item.showZero ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
                    }`} />
                  )}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>

        {/* Export / Import dropdown */}
        <div ref={backupMenuRef} className="relative px-2 py-2 border-t border-zinc-800/60">
          {restoreStatus && (
            <div className="mb-1 text-[10px] text-emerald-400 px-1">{restoreStatus}</div>
          )}
          <button
            onClick={() => setShowBackupMenu(!showBackupMenu)}
            className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 transition-colors"
            title="Export / Import State"
          >
            <span className="w-4 text-center text-zinc-500 shrink-0">⇅</span>
            {!collapsed && <span className="flex-1 text-left">Export / Import State</span>}
            {!collapsed && (
              <svg className="w-3 h-3 text-zinc-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            )}
          </button>
          {showBackupMenu && (
            <div className={`absolute ${collapsed ? 'left-full ml-1' : 'left-2 right-2'} bottom-full mb-1 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl z-50 min-w-[10rem]`}>
              <button
                onClick={handleDownloadBackup}
                className="w-full text-left px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 rounded-t-lg transition-colors"
              >
                Export instance
              </button>
              <button
                onClick={handleRestoreBackup}
                className="w-full text-left px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 rounded-b-lg transition-colors border-t border-zinc-800"
              >
                Import instance…
              </button>
            </div>
          )}
        </div>

        {/* Footer: just the version. Connected clients are now a top-level
            sidebar item (Manage → Connected) with a live badge count. */}
        {!collapsed && (
          <div className="px-3 py-2 border-t border-zinc-800/60 text-[10px] text-zinc-600">
            {appVersion ? `v${appVersion}` : '…'}
          </div>
        )}
      </aside>

      {/* Main content — pages handle their own padding and max-width.
          Fullscreen pages (Studio, Intents, Router) use `h-full`
          and fill the viewport. Content pages add `p-6 max-w-*` themselves. */}
      <main className="flex-1 min-h-0 min-w-0 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
