import { NavLink, Outlet } from 'react-router-dom';
import { useAppStore } from '@/store';
import { useState, useEffect } from 'react';
import { api } from '@/api/client';

const links = [
  { to: '/', label: 'Playground' },
  { to: '/intents', label: 'Intents' },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/scenarios', label: 'Training' },
  { to: '/discovery', label: 'Discovery' },
  { to: '/review', label: 'Review' },
  { to: '/apps', label: 'Apps' },
  { to: '/import', label: 'Import' },
  { to: '/settings', label: 'Settings' },
];

export default function Layout() {
  const { settings, setMode, setSelectedAppId } = useAppStore();
  const isLearn = settings.mode === 'learn';
  const [apps, setApps] = useState<string[]>(['default']);
  const [showAppMenu, setShowAppMenu] = useState(false);
  const [newAppName, setNewAppName] = useState('');

  useEffect(() => {
    api.listApps().then(setApps).catch(() => {});
  }, []);

  const handleCreateApp = async () => {
    const name = newAppName.trim().toLowerCase().replace(/\s+/g, '-');
    if (!name) return;
    try {
      await api.createApp(name);
      setApps(prev => [...prev, name]);
      setSelectedAppId(name);
      setNewAppName('');
      setShowAppMenu(false);
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Failed to create app');
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <nav className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 flex items-center h-12 gap-6">
          <span className="font-semibold text-white tracking-tight">ASV Router</span>
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

          {/* Right side: app selector + mode toggle */}
          <div className="ml-auto flex items-center gap-3">
            {/* App selector */}
            <div className="relative">
              <button
                onClick={() => setShowAppMenu(!showAppMenu)}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-zinc-800 text-zinc-300 hover:text-white border border-zinc-700 transition-colors"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                {settings.selectedAppId}
                <svg className="w-3 h-3 ml-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
              </button>

              {showAppMenu && (
                <div className="absolute right-0 top-full mt-1 w-56 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl z-50 py-1">
                  {apps.map(app => (
                    <button
                      key={app}
                      onClick={() => { setSelectedAppId(app); setShowAppMenu(false); window.location.reload(); }}
                      className={`w-full text-left px-3 py-1.5 text-sm hover:bg-zinc-800 transition-colors ${
                        app === settings.selectedAppId ? 'text-blue-400 font-medium' : 'text-zinc-300'
                      }`}
                    >
                      {app}
                    </button>
                  ))}
                  <div className="border-t border-zinc-700 mt-1 pt-1 px-2 pb-1">
                    <div className="flex gap-1">
                      <input
                        type="text"
                        value={newAppName}
                        onChange={e => setNewAppName(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && handleCreateApp()}
                        placeholder="New app name..."
                        className="flex-1 bg-zinc-800 text-xs text-white px-2 py-1 rounded border border-zinc-600 focus:outline-none focus:border-blue-500"
                      />
                      <button
                        onClick={handleCreateApp}
                        className="text-xs px-2 py-1 bg-blue-600 text-white rounded hover:bg-blue-500"
                      >
                        +
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Mode toggle */}
            <button
              onClick={() => setMode(isLearn ? 'production' : 'learn')}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold transition-colors border ${
                isLearn
                  ? 'bg-amber-400/15 text-amber-400 border-amber-400/30'
                  : 'bg-emerald-400/15 text-emerald-400 border-emerald-400/30'
              }`}
              title={isLearn ? 'Learn mode: LLM reviews every query' : 'Production mode: fast routing only'}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${isLearn ? 'bg-amber-400' : 'bg-emerald-400'}`} />
              {isLearn ? 'Learn' : 'Production'}
            </button>
          </div>
        </div>
      </nav>
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
