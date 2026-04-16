import { NavLink, Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useAppStore } from '@/store';
import { useState, useEffect } from 'react';
import { api } from '@/api/client';

const links = [
  { to: '/namespaces',   label: 'Namespaces' },
  { to: '/intents',      label: 'Intents' },
  { to: '/studio',       label: 'Studio' },
  { to: '/build',        label: 'Build' },
  { to: '/import',       label: 'Import' },
  { to: '/settings',     label: 'Settings' },
];

export default function Layout() {
  const { settings, setSelectedNamespaceId, setSelectedDomain } = useAppStore();
  const navigate = useNavigate();
  const location = useLocation();
  const fullscreen = location.pathname.startsWith('/build');

  const [namespaces, setNamespaces] = useState<string[]>(['default']);
  const [domains, setDomains] = useState<string[]>([]);
  const [showNsMenu, setShowNsMenu] = useState(false);
  const [showDomainMenu, setShowDomainMenu] = useState(false);

  useEffect(() => {
    api.listNamespaces().then(ns => setNamespaces(ns.map(n => n.id))).catch(() => {});
  }, []);

  // Reload domains when namespace changes
  useEffect(() => {
    api.listDomains().then(ds => setDomains(ds.map(d => d.name))).catch(() => setDomains([]));
  }, [settings.selectedNamespaceId]);

  const switchNamespace = (namespaceId: string) => {
    setSelectedNamespaceId(namespaceId);
    setSelectedDomain('');
    setShowNsMenu(false);
    window.location.reload();
  };

  const switchDomain = (domain: string) => {
    setSelectedDomain(domain);
    setShowDomainMenu(false);
    navigate('/intents');
  };

  const activeNs = settings.selectedNamespaceId;
  const activeDomain = settings.selectedDomain;

  return (
    <div className={`${fullscreen ? 'h-screen overflow-hidden' : 'min-h-screen'} flex flex-col`}>
      <nav className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur sticky top-0 z-50">
        <div className={`${fullscreen ? 'px-4' : 'max-w-7xl mx-auto px-4'} flex items-center h-12 gap-6`}>
          <NavLink to="/" className="font-semibold text-white tracking-tight hover:text-violet-400 transition-colors">
            ASV Router
          </NavLink>

          <div className="flex gap-1">
            {links.map(l => (
              <NavLink
                key={l.to}
                to={l.to}
                className={({ isActive }) =>
                  `px-3 py-1.5 text-sm rounded-md transition-colors ${
                    isActive
                      ? 'bg-zinc-800 text-white'
                      : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
                  }`
                }
              >
                {l.label}
              </NavLink>
            ))}
          </div>

          {/* Right side */}
          <div className="ml-auto flex items-center gap-1">

            {/* Namespace selector */}
            <div className="relative">
              <button
                onClick={() => { setShowNsMenu(!showNsMenu); setShowDomainMenu(false); }}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-zinc-800 text-zinc-300 hover:text-white border border-zinc-700 transition-colors"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                {activeNs}
                <svg className="w-3 h-3 ml-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {showNsMenu && (
                <div className="absolute right-0 top-full mt-1 w-56 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl z-50 py-1">
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
                      + Manage namespaces
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Domain selector — only shown when namespace has domains */}
            {domains.length > 0 && (
              <>
                <span className="text-zinc-600 text-xs select-none">/</span>
                <div className="relative">
                  <button
                    onClick={() => { setShowDomainMenu(!showDomainMenu); setShowNsMenu(false); }}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-zinc-900 text-zinc-400 hover:text-white border border-zinc-700/60 transition-colors"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-violet-400/60" />
                    {activeDomain ? activeDomain : 'all domains'}
                    <svg className="w-3 h-3 ml-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {showDomainMenu && (
                    <div className="absolute right-0 top-full mt-1 w-52 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl z-50 py-1">
                      <button
                        onClick={() => switchDomain('')}
                        className={`w-full text-left px-3 py-1.5 text-sm hover:bg-zinc-800 transition-colors ${
                          !activeDomain ? 'text-violet-400 font-medium' : 'text-zinc-300'
                        }`}
                      >
                        All domains
                      </button>
                      <div className="border-t border-zinc-800 my-1" />
                      {domains.map(d => (
                        <button
                          key={d}
                          onClick={() => switchDomain(d)}
                          className={`w-full text-left px-3 py-1.5 text-sm font-mono hover:bg-zinc-800 transition-colors ${
                            d === activeDomain ? 'text-violet-400 font-medium' : 'text-zinc-300'
                          }`}
                        >
                          {d}
                        </button>
                      ))}
                      <div className="border-t border-zinc-700 mt-1 pt-1 px-2 pb-1">
                        <button
                          onClick={() => { setShowDomainMenu(false); navigate('/namespaces'); }}
                          className="w-full text-left text-xs text-zinc-400 hover:text-violet-400 px-1 py-1"
                        >
                          Manage domains →
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}

          </div>
        </div>
      </nav>

      <main className={fullscreen ? 'flex-1 min-h-0' : 'flex-1 max-w-7xl mx-auto w-full px-4 py-6'}>
        <Outlet />
      </main>
    </div>
  );
}
