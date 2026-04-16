import type { ReactNode } from 'react';

/**
 * Shared wrapper for content-style pages (Namespaces, Domains, Debug, etc).
 * Pages wanting fullscreen (Studio, Intents, Router) use `h-full` directly
 * and don't use this wrapper.
 */
type Size = 'sm' | 'md' | 'lg' | 'xl' | 'full';

const MAX_W: Record<Size, string> = {
  sm:   'max-w-3xl',   // forms, small lists
  md:   'max-w-5xl',   // medium content
  lg:   'max-w-6xl',   // wide tables, dashboards
  xl:   'max-w-7xl',   // very wide
  full: 'max-w-none',  // no cap
};

export default function PageContainer({
  children,
  size = 'sm',
  className = '',
}: {
  children: ReactNode;
  size?: Size;
  className?: string;
}) {
  return (
    <div className={`${MAX_W[size]} mx-auto p-6 ${className}`}>
      {children}
    </div>
  );
}
