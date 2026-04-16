import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useAppStore } from '@/store';
import { useState, useEffect } from 'react';
import { api } from '@/api/client';

// 2026 layout: sidebar + fullscreen main area. No top navbar bloat.
// Namespace switcher lives in the sidebar header. Domains are managed
// inside NamespacesPage/DomainsPage, not global nav.

type NavItem = {
  to: string;
  label: string;
  icon: string;
  hint?: string;
};

const NAV_ITEMS: NavItem[] = [
  { to: '/',           label: 'Route',     icon: '▸', hint: 'Test queries' },
  { to: '/intents',    label: 'Intents',   icon: '◆', hint: 'Manage intents' },
  { to: '/studio',     label: 'Studio',    icon: '◉', hint: 'Training + review' },
  { to: '/import',     label: 'Import',    icon: '↓', hint: 'OpenAPI / MCP' },
  { to: '/namespaces', label: 'Workspaces', icon: '▦', hint: 'Manage namespaces' },
  { to: '/settings',   label: 'Settings',  icon: '⚙', hint: 'Config' },
];

export default function Layout() {
  const { settings, setSelectedNamespaceId, setSelectedDomain } = useAppStore();
  const navigate = useNavigate();

  const [namespaces, setNamespaces] = useState<string[]>(['default']);
  const [showNsMenu, setShowNsMenu] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    api.listNamespaces().then(ns => setNamespaces(ns.map(n => n.id))).catch(() => {});
  }, []);

  const switchNamespace = (namespaceId: string) => {
    setSelectedNamespaceId(namespaceId);
    setSelectedDomain('');
    setShowNsMenu(false);
    window.location.reload();
  };

  const activeNs = settings.selectedNamespaceId;

  const sidebarWidth = collapsed ? 'w-14' : 'w-56';

  return (
    <div className="h-screen flex overflow-hidden bg-zinc-950">
      {/* Sidebar */}
      <aside className={`${sidebarWidth} shrink-0 flex flex-col border-r border-zinc-800 bg-zinc-950 transition-all duration-150`}>
        {/* Brand + collapse */}
        <div className="h-12 flex items-center px-3 border-b border-zinc-800">
          {!collapsed && (
            <span className="text-sm font-semibold text-white tracking-tight">ASV</span>
          )}
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
        <div className="px-2 py-2 border-b border-zinc-800/60 relative">
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
            <div className={`absolute ${collapsed ? 'left-full ml-1' : 'left-2 right-2'} top-full mt-1 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl z-50 py-1 min-w-[12rem]`}>
              {namespaces.map(ns => (
                <button
                  key={ns}
                  onClick={() => switchNamespace(ns)}
                  className={`w-full text-left px-3 py-1.5 text-sm hover:bg-zinc-800 transition-colors ${
                    ns === activeNs ? 'text-blue-400 font-medium' : 'text-zinc-300'
                  }`}
                >
                  {ns}
                </button>
              ))}
              <div className="border-t border-zinc-700 mt-1 pt-1 px-2 pb-1">
                <button
                  onClick={() => { setShowNsMenu(false); navigate('/namespaces'); }}
                  className="w-full text-left text-xs text-zinc-400 hover:text-violet-400 px-1 py-1"
                >
                  + Manage workspaces
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Nav items */}
        <nav className="flex-1 py-2 overflow-y-auto">
          {NAV_ITEMS.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `mx-2 my-0.5 px-2 py-1.5 rounded flex items-center gap-2.5 text-sm transition-colors ${
                  isActive
                    ? 'bg-zinc-800 text-white'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
                }`
              }
              title={collapsed ? item.label : item.hint}
            >
              <span className="w-4 text-center text-zinc-500 shrink-0">{item.icon}</span>
              {!collapsed && <span className="truncate">{item.label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        {!collapsed && (
          <div className="px-3 py-2 border-t border-zinc-800/60 text-[10px] text-zinc-600">
            v0.1 · sub-ms routing
          </div>
        )}
      </aside>

      {/* Main content. Default: padded with a sensible max-width.
          Pages wanting fullscreen (Studio) wrap their content in `h-full`
          and use their own internal layout. */}
      <main className="flex-1 min-h-0 min-w-0 overflow-auto">
        <div className="min-h-full p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
