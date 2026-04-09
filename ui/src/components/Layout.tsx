import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useAppStore } from '@/store';
import { useState, useEffect } from 'react';
import { api } from '@/api/client';

const links = [
  { to: '/intents', label: 'Intents' },
  { to: '/review', label: 'Review' },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/scenarios', label: 'Training' },
  { to: '/discovery', label: 'Discovery' },
  { to: '/settings', label: 'Settings' },
];

export default function Layout() {
  const { settings, setSelectedAppId } = useAppStore();
  const [apps, setApps] = useState<string[]>(['default']);
  const [showAppMenu, setShowAppMenu] = useState(false);

  useEffect(() => {
    api.listApps().then(setApps).catch(() => {});
  }, []);


  return (
    <div className="min-h-screen flex flex-col">
      <nav className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 flex items-center h-12 gap-6">
          <NavLink to="/" className="font-semibold text-white tracking-tight hover:text-violet-400 transition-colors">ASV Router</NavLink>
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
                    <button
                      onClick={() => { setShowAppMenu(false); window.location.href = '/settings'; }}
                      className="w-full text-left text-xs text-zinc-400 hover:text-violet-400 px-1 py-1"
                    >
                      + Manage Apps
                    </button>
                  </div>
                </div>
              )}
            </div>

          </div>
        </div>
      </nav>
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
