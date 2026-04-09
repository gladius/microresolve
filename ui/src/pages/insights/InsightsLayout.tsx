import { NavLink, Outlet, useLocation } from 'react-router-dom';

const sections = [
  { to: '/insights', label: 'Overview', end: true },
  { to: '/insights/discovery', label: 'Discovery' },
  { to: '/insights/projections', label: 'Projections' },
  { to: '/insights/workflows', label: 'Workflows' },
  { to: '/insights/temporal', label: 'Temporal Flow' },
  { to: '/insights/escalations', label: 'Escalations' },
  { to: '/insights/cooccurrence', label: 'Co-occurrence' },
];

export default function InsightsLayout() {
  return (
    <div className="flex gap-0 h-[calc(100vh-6rem)] -mx-4">
      {/* Sidebar */}
      <div className="w-48 min-w-[12rem] border-r border-zinc-800 flex flex-col">
        <div className="px-3 py-3 border-b border-zinc-800">
          <span className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">Insights</span>
        </div>
        <div className="flex-1 overflow-y-auto">
          {sections.map(s => (
            <NavLink
              key={s.to}
              to={s.to}
              end={s.end}
              className={({ isActive }) =>
                `block px-3 py-2 text-sm transition-colors border-l-2 ${
                  isActive
                    ? 'text-white bg-zinc-800/50 border-violet-400'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/30 border-transparent'
                }`
              }
            >
              {s.label}
            </NavLink>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-5">
        <Outlet />
      </div>
    </div>
  );
}
