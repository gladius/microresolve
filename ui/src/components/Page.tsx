import type { ReactNode } from 'react';

/**
 * Universal page shell. Every page uses this.
 *
 * - Always renders a 48px top bar aligned with the main sidebar's brand header
 *   (seamless horizontal line across the app chrome).
 * - Content area fills the remaining viewport, scrollable.
 * - Two content modes:
 *     fullscreen: content flows edge-to-edge (default for Studio/Intents/Router)
 *     contained:  content wrapped in max-width + padding (for Settings, Namespaces, etc)
 *
 * Usage:
 *   <Page title="Namespaces" subtitle="Isolated workspaces" actions={<Button>New</Button>} size="sm">
 *     <MyContent />
 *   </Page>
 *
 *   <Page title="Studio" fullscreen>
 *     <MyComplexIDELayout />
 *   </Page>
 */
type Size = 'sm' | 'md' | 'lg' | 'xl' | 'full';

const MAX_W: Record<Size, string> = {
  sm:   'max-w-3xl',
  md:   'max-w-5xl',
  lg:   'max-w-6xl',
  xl:   'max-w-7xl',
  full: 'max-w-none',
};

interface PageProps {
  children: ReactNode;
  title?: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
  /** If true, content area is edge-to-edge with no max-width / padding. */
  fullscreen?: boolean;
  /** Max-width preset for contained mode. Default 'md'. Ignored if fullscreen. */
  size?: Size;
  /** Extra className applied to the content wrapper (only in contained mode). */
  className?: string;
}

export default function Page({
  children,
  title,
  subtitle,
  actions,
  fullscreen = false,
  size = 'md',
  className = '',
}: PageProps) {
  return (
    <div className="h-full flex flex-col">
      {/* Top bar — 48px, aligned with sidebar brand header */}
      <div className="h-12 flex items-center gap-3 px-6 border-b border-zinc-800 shrink-0 bg-zinc-950/40">
        {title && <h1 className="text-sm font-semibold text-zinc-100">{title}</h1>}
        {subtitle && <span className="text-xs text-zinc-500">{subtitle}</span>}
        {actions && <div className="ml-auto flex items-center gap-2">{actions}</div>}
      </div>

      {/* Content */}
      {fullscreen ? (
        <div className="flex-1 min-h-0 overflow-auto">{children}</div>
      ) : (
        <div className="flex-1 overflow-auto">
          <div className={`${MAX_W[size]} mx-auto p-6 ${className}`}>
            {children}
          </div>
        </div>
      )}
    </div>
  );
}
