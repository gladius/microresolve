import { type ReactNode } from 'react';

export interface SidebarItem {
  id: string;
  label: string;
  badge?: string | number;
  color?: string;
}

interface SidebarLayoutProps {
  title: string;
  items: SidebarItem[];
  selected: string | null;
  onSelect: (id: string) => void;
  headerActions?: ReactNode;
  children: ReactNode;
}

export default function SidebarLayout({
  title, items, selected, onSelect, headerActions, children,
}: SidebarLayoutProps) {
  return (
    <div className="flex gap-0 h-full">
      {/* Sub-sidebar */}
      <div className="w-56 min-w-[14rem] border-r border-zinc-800 flex flex-col">
        <div className="h-12 px-4 border-b border-zinc-800 flex items-center justify-between flex-shrink-0">
          <span className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">{title}</span>
          {headerActions}
        </div>
        <div className="flex-1 overflow-y-auto">
          {items.map(item => (
            <div
              key={item.id}
              onClick={() => onSelect(item.id)}
              className={`px-3 py-2 cursor-pointer border-b border-zinc-800/50 transition-colors flex items-center justify-between ${
                selected === item.id ? 'bg-zinc-800/80' : 'hover:bg-zinc-800/40'
              }`}
            >
              <span className={`text-sm ${
                selected === item.id ? 'text-zinc-100 font-medium' : 'text-zinc-400'
              }`}>
                {item.label}
              </span>
              {item.badge !== undefined && (
                <span className={`text-[10px] ${item.color || 'text-zinc-600'}`}>
                  {item.badge}
                </span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {children}
      </div>
    </div>
  );
}
