import { NavLink, Outlet } from 'react-router-dom';
import { useAppStore } from '@/store';

const links = [
  { to: '/', label: 'Router' },
  { to: '/intents', label: 'Intents' },
  { to: '/projections', label: 'Projections' },
  { to: '/scenarios', label: 'Training' },
  { to: '/debug', label: 'Debug' },
  { to: '/settings', label: 'Settings' },
];

export default function Layout() {
  const { settings, setMode } = useAppStore();
  const isLearn = settings.mode === 'learn';

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

          {/* Mode toggle — right side of navbar */}
          <div className="ml-auto flex items-center gap-2">
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
