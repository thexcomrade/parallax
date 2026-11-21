import React, { useEffect } from 'react';
import { NavLink, Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  ScanEye,
  History,
  FolderArchive,
  Settings,
  ChevronLeft,
  ChevronRight,
  X,
  type LucideIcon,
} from 'lucide-react';
import { clsx } from 'clsx';
import { ParallaxLogo } from '@/components/brand/ParallaxLogo';

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  exact?: boolean;
  matchPrefix?: string;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

const NAV_SECTIONS: NavSection[] = [
  {
    title: 'Bench',
    items: [
      { to: '/app', label: 'Dashboard', icon: LayoutDashboard, exact: true },
      { to: '/app/analyze', label: 'Analyze', icon: ScanEye, matchPrefix: '/app/analyses' },
      { to: '/app/history', label: 'History', icon: History },
    ],
  },
  {
    title: 'Corpus',
    items: [{ to: '/app/library', label: 'Library', icon: FolderArchive }],
  },
  {
    title: 'Operator',
    items: [{ to: '/app/settings', label: 'Settings', icon: Settings }],
  },
];

const MOBILE_ITEMS: NavItem[] = [
  { to: '/app', label: 'Dashboard', icon: LayoutDashboard, exact: true },
  { to: '/app/analyze', label: 'Analyze', icon: ScanEye, matchPrefix: '/app/analyses' },
  { to: '/app/history', label: 'History', icon: History },
  { to: '/app/library', label: 'Library', icon: FolderArchive },
  { to: '/app/settings', label: 'Settings', icon: Settings },
];

export interface CommandRailProps {
  collapsed?: boolean;
  onToggleCollapse?: () => void;
  mobileOpen?: boolean;
  onCloseMobile?: () => void;
}

export const CommandRail: React.FC<CommandRailProps> = ({
  collapsed = false,
  onToggleCollapse,
  mobileOpen = false,
  onCloseMobile,
}) => {
  const location = useLocation();

  useEffect(() => {
    if (!mobileOpen) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onCloseMobile?.();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [mobileOpen, onCloseMobile]);

  const isItemActive = (item: NavItem): boolean => {
    if (item.exact) return location.pathname === item.to;
    if (item.matchPrefix && location.pathname.startsWith(item.matchPrefix)) return true;
    return location.pathname.startsWith(item.to);
  };

  const showLabels = !collapsed || mobileOpen;

  return (
    <>
      {mobileOpen && (
        <button
          type="button"
          tabIndex={-1}
          aria-label="Close navigation menu"
          onClick={onCloseMobile}
          className="md:hidden fixed inset-0 z-40 bg-black/70 cursor-default"
        />
      )}

      <aside
        aria-label="Application navigation"
        className={clsx(
          'top-0 bottom-0 left-0 z-50 flex flex-col h-screen select-none shrink-0',
          'bg-[var(--surface-card)] border-r border-[var(--line)]',
          'md:sticky md:z-20 transition-[width] duration-[var(--motion-normal)] ease-[var(--ease-instrument)] motion-reduce:transition-none',
          collapsed ? 'md:w-[72px] md:min-w-[72px] md:max-w-[72px]' : 'md:w-[260px] md:min-w-[260px] md:max-w-[260px]',
          mobileOpen ? 'fixed w-[260px] max-w-[85vw] shadow-[var(--shadow-overlay)] rail-in' : 'max-md:hidden',
        )}
      >
        {/* Brand header — aligns with the 72px global status strip */}
        <div
          className={clsx(
            'h-[72px] border-b border-[var(--line)] flex items-center shrink-0',
            collapsed ? 'md:justify-center md:px-0 px-4' : 'px-4 gap-3',
          )}
        >
          <Link
            to="/app"
            onClick={onCloseMobile}
            title={collapsed ? 'PARALLAX — dashboard' : undefined}
            className={clsx(
              'flex items-center rounded-[var(--radius-sm)] text-[var(--cobalt-300)]',
              collapsed ? 'md:justify-center' : 'gap-3',
            )}
          >
            <span className="w-9 h-9 rounded-[var(--radius-lg)] bg-[var(--surface-base)] border border-[var(--line)] flex items-center justify-center shrink-0">
              <ParallaxLogo size={22} />
            </span>

            {showLabels && (
              <span className={clsx('flex flex-col min-w-0', collapsed && 'md:hidden')}>
                <span className="font-display text-[15px] font-bold tracking-[0.08em] text-[var(--text-100)] leading-tight">
                  PARALLAX
                </span>
                <span className="font-mono text-[10px] text-[var(--text-400)] leading-tight mt-0.5 truncate">
                  AVT²-DWF FORENSICS
                </span>
              </span>
            )}
          </Link>

          <button
            type="button"
            onClick={onCloseMobile}
            aria-label="Close navigation menu"
            className="md:hidden ml-auto w-9 h-9 rounded-[var(--radius-md)] border border-[var(--line)] text-[var(--text-300)] hover:text-[var(--text-100)] hover:border-[var(--line-strong)] flex items-center justify-center interactive-btn"
          >
            <X size={17} />
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto px-2.5 py-4 flex flex-col">
          {NAV_SECTIONS.map((section, sectionIdx) => (
            <div key={section.title} className={clsx('flex flex-col', sectionIdx > 0 && 'pt-5')}>
              {showLabels && (
                <div
                  className={clsx(
                    'px-3 mb-1.5 font-mono text-[10px] tracking-[0.12em] uppercase text-[var(--text-400)]',
                    collapsed && 'md:hidden',
                  )}
                >
                  {section.title}
                </div>
              )}

              <ul className="flex flex-col gap-1 list-none">
                {section.items.map((item) => {
                  const Icon = item.icon;
                  const active = isItemActive(item);

                  return (
                    <li key={item.to} className="relative">
                      <NavLink
                        to={item.to}
                        onClick={onCloseMobile}
                        aria-label={collapsed ? item.label : undefined}
                        className={clsx(
                          'group relative flex items-center rounded-[var(--radius-md)] text-[13px] font-medium interactive-btn',
                          collapsed ? 'md:justify-center md:px-0 h-10 px-3 gap-3' : 'h-10 px-3 gap-3',
                          active
                            ? 'bg-[var(--surface-hover)] text-[var(--text-100)]'
                            : 'text-[var(--text-300)] hover:bg-[var(--surface-hover)]/60 hover:text-[var(--text-100)]',
                        )}
                      >
                        <span
                          aria-hidden="true"
                          className={clsx(
                            'absolute left-0 top-1/2 -translate-y-1/2 h-4 w-0.5 rounded-full transition-colors',
                            active ? 'bg-[var(--cobalt-400)]' : 'bg-transparent',
                          )}
                        />
                        <Icon
                          size={17}
                          strokeWidth={1.75}
                          className={clsx(
                            'shrink-0 transition-colors',
                            active
                              ? 'text-[var(--cobalt-300)]'
                              : 'text-[var(--text-400)] group-hover:text-[var(--text-200)]',
                          )}
                        />
                        {showLabels && (
                          <span className={clsx('truncate flex-1', collapsed && 'md:hidden')}>
                            {item.label}
                          </span>
                        )}
                      </NavLink>

                      {collapsed && (
                        <span
                          role="tooltip"
                          className="hidden md:block absolute left-full top-1/2 -translate-y-1/2 ml-2 px-2 py-1 rounded-[var(--radius-sm)] bg-[var(--surface-overlay)] border border-[var(--line-strong)] text-[11px] text-[var(--text-100)] whitespace-nowrap opacity-0 pointer-events-none group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity duration-[var(--motion-fast)] z-50"
                        >
                          {item.label}
                        </span>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </nav>

        {onToggleCollapse && (
          <div className="hidden md:flex items-center p-2.5 border-t border-[var(--line)] shrink-0">
            <button
              type="button"
              onClick={onToggleCollapse}
              aria-expanded={!collapsed}
              aria-label={collapsed ? 'Expand rail' : 'Collapse rail'}
              className={clsx(
                'group flex items-center h-9 rounded-[var(--radius-md)] text-[12px] font-medium text-[var(--text-400)] hover:text-[var(--text-100)] hover:bg-[var(--surface-hover)] interactive-btn',
                collapsed ? 'w-full justify-center' : 'w-full justify-between px-3',
              )}
            >
              {!collapsed && <span>Collapse rail</span>}
              {collapsed ? <ChevronRight size={15} /> : <ChevronLeft size={15} />}
            </button>
          </div>
        )}
      </aside>

      {/* Mobile quick bar — solid surface, no blur, 44px touch targets. */}
      <nav
        aria-label="Mobile navigation"
        className="md:hidden fixed bottom-0 left-0 right-0 h-14 bg-[var(--surface-card)] border-t border-[var(--line)] flex items-center justify-around px-1 z-30 select-none"
      >
        {MOBILE_ITEMS.map((item) => {
          const Icon = item.icon;
          const active = isItemActive(item);

          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={clsx(
                'flex flex-col items-center justify-center min-w-[52px] min-h-[44px] rounded-[var(--radius-md)] font-mono text-[9px] uppercase tracking-wide',
                active ? 'text-[var(--cobalt-300)]' : 'text-[var(--text-400)]',
              )}
            >
              <Icon size={17} strokeWidth={active ? 2.25 : 1.75} />
              <span className="truncate mt-1 leading-none">{item.label}</span>
            </NavLink>
          );
        })}
      </nav>
    </>
  );
};
