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
  Shield,
  X,
  type LucideIcon,
} from 'lucide-react';
import { clsx } from 'clsx';

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  badge?: string;
  exact?: boolean;
  matchPrefix?: string;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

const NAV_SECTIONS: NavSection[] = [
  {
    title: 'MAIN',
    items: [
      { to: '/app', label: 'Dashboard', icon: LayoutDashboard, exact: true },
      { to: '/app/analyze', label: 'Analyze', icon: ScanEye, matchPrefix: '/app/analyses' },
      { to: '/app/history', label: 'History', icon: History },
    ],
  },
  {
    title: 'WORKSPACE',
    items: [
      { to: '/app/library', label: 'Library', icon: FolderArchive },
    ],
  },
  {
    title: 'PREFERENCES',
    items: [
      { to: '/app/settings', label: 'Settings', icon: Settings },
    ],
  },
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

  // Handle Escape key to dismiss mobile drawer
  useEffect(() => {
    if (!mobileOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onCloseMobile?.();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [mobileOpen, onCloseMobile]);

  // Determine active state including nested routes
  const isItemActive = (item: NavItem) => {
    if (item.exact) {
      return location.pathname === item.to;
    }
    if (item.matchPrefix && location.pathname.startsWith(item.matchPrefix)) {
      return true;
    }
    return location.pathname.startsWith(item.to);
  };

  return (
    <>
      {/* Mobile Drawer Overlay Backdrop */}
      {mobileOpen && (
        <div
          role="presentation"
          onClick={onCloseMobile}
          className="md:hidden fixed inset-0 z-40 bg-black/60 backdrop-blur-xs transition-opacity animate-in fade-in duration-200"
        />
      )}

      {/* Main Sidebar: Desktop Fixed/Sticky Full-Height + Mobile Slide-Out Drawer */}
      <aside
        aria-label="Application Navigation"
        className={clsx(
          'top-0 bottom-0 left-0 z-50 flex flex-col bg-[#0D131E] border-r border-[#253247] select-none shrink-0 h-screen',
          // Desktop styling: 260px expanded, 72px collapsed, sticky full-viewport height
          'md:sticky md:z-20 transition-[width] duration-200 ease-out motion-reduce:transition-none',
          collapsed ? 'md:w-[72px] md:min-w-[72px] md:max-w-[72px]' : 'md:w-[260px] md:min-w-[260px] md:max-w-[260px]',
          // Mobile state: slide-in drawer
          mobileOpen
            ? 'fixed w-[260px] max-w-[85vw] shadow-2xl animate-in slide-in-from-left duration-200'
            : 'max-md:hidden',
        )}
      >
        {/* ── Brand Header (Height: 72px, matching Global Header horizon) ── */}
        <div
          className={clsx(
            'h-[72px] border-b border-[#253247] flex items-center shrink-0 bg-[#0D131E]',
            collapsed ? 'md:justify-center md:px-0 px-5' : 'px-5 gap-3.5',
          )}
        >
          <Link
            to="/app"
            onClick={onCloseMobile}
            title={collapsed ? 'PARALLAX Forensics' : undefined}
            className={clsx(
              'flex items-center group focus-visible:outline-2 focus-visible:outline-[#648CFF] rounded-lg',
              collapsed ? 'md:justify-center' : 'gap-3.5',
            )}
          >
            {/* Logo Container (40 × 40px) */}
            <div className="w-10 h-10 rounded-xl bg-[#111827] border border-[#253247] group-hover:border-[#648CFF]/50 transition-colors flex items-center justify-center text-[#648CFF] shadow-xs shrink-0">
              <Shield size={20} strokeWidth={2} />
            </div>

            {/* Brand Name & Subtitle */}
            {(!collapsed || mobileOpen) && (
              <div className={clsx('flex flex-col min-w-0', collapsed && 'md:hidden')}>
                <span className="text-[16px] font-semibold text-[#E8EDF5] tracking-wide leading-tight">
                  PARALLAX
                </span>
                <span className="text-[12px] text-[#91A0B8] font-normal leading-tight mt-0.5 truncate">
                  Deepfake Detection System
                </span>
              </div>
            )}
          </Link>

          {/* Close Control for Mobile Drawer */}
          <button
            type="button"
            onClick={onCloseMobile}
            aria-label="Close navigation menu"
            className="md:hidden ml-auto w-8 h-8 rounded-lg bg-[#111827] border border-[#253247] hover:border-[#648CFF]/40 text-[#91A0B8] hover:text-[#E8EDF5] flex items-center justify-center transition-colors focus-visible:outline-2 focus-visible:outline-[#648CFF]"
          >
            <X size={17} />
          </button>
        </div>

        {/* ── Navigation Sections (Middle: MAIN, WORKSPACE, PREFERENCES) ── */}
        <div className="flex-1 overflow-y-auto px-3 py-4 flex flex-col">
          {NAV_SECTIONS.map((section, sectionIdx) => (
            <div
              key={section.title}
              className={clsx('flex flex-col', sectionIdx > 0 ? 'pt-6' : 'pt-1')}
            >
              {/* Section Label (11px, font-weight 600, WCAG AA compliant slate, aligned with nav labels) */}
              {!collapsed && (
                <div className="px-3 mb-1.5 text-[11px] font-semibold tracking-wider text-[#7E8B9F] uppercase">
                  {section.title}
                </div>
              )}

              {/* Navigation Items (44px height, 12px padding, 19px icon, 12px gap, 9px radius, 4-6px vertical gap) */}
              <nav aria-label={section.title} className="flex flex-col gap-1.5">
                {section.items.map((item) => {
                  const Icon = item.icon;
                  const active = isItemActive(item);

                  return (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      onClick={onCloseMobile}
                      className={clsx(
                        'group relative flex items-center rounded-[9px] text-[13px] font-medium transition-all duration-150 focus-visible:outline-2 focus-visible:outline-[#648CFF] focus-visible:outline-offset-1',
                        collapsed
                          ? 'md:justify-center md:px-0 h-[44px] px-3 gap-3'
                          : 'h-[44px] px-3 gap-3',
                        active
                          ? 'bg-[#192437] text-[#E8EDF5] border border-[#2B3B55]'
                          : 'text-[#91A0B8] bg-transparent border border-transparent hover:bg-[#151D2A] hover:text-[#E8EDF5] hover:border-[#253247]/60',
                      )}
                    >
                      <Icon
                        size={19}
                        strokeWidth={2}
                        className={clsx(
                          'shrink-0 transition-colors',
                          active
                            ? 'text-[#648CFF]'
                            : 'text-[#91A0B8] group-hover:text-[#E8EDF5]',
                        )}
                      />

                      {/* Label Text (Sentence case, 13px, font-medium) */}
                      {(!collapsed || mobileOpen) && (
                        <span className={clsx('truncate flex-1 leading-none', collapsed && 'md:hidden')}>
                          {item.label}
                        </span>
                      )}

                      {(!collapsed || mobileOpen) && item.badge && (
                        <span className={clsx('font-mono text-[10px] px-1.5 py-0.5 rounded bg-[#171F2D] text-[#648CFF] border border-[#253247]', collapsed && 'md:hidden')}>
                          {item.badge}
                        </span>
                      )}

                      {/* Accessible Tooltip in Collapsed Desktop Mode */}
                      {collapsed && (
                        <div
                          role="tooltip"
                          className="hidden md:block absolute left-full ml-3 px-3 py-1.5 rounded-lg bg-[#171F2D] border border-[#253247] text-xs font-medium text-[#E8EDF5] shadow-xl whitespace-nowrap opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity duration-150 z-50"
                        >
                          {item.label}
                        </div>
                      )}
                    </NavLink>
                  );
                })}
              </nav>
            </div>
          ))}
        </div>

        {/* ── Footer: Sidebar Collapse / Expand Toggle (Desktop Only) ── */}
        {onToggleCollapse && (
          <div className="hidden md:flex items-center p-3 border-t border-[#253247] bg-[#0D131E] shrink-0">
            <button
              type="button"
              onClick={onToggleCollapse}
              aria-expanded={!collapsed}
              aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              className={clsx(
                'group relative flex items-center rounded-[9px] text-[13px] font-medium text-[#91A0B8] hover:text-[#E8EDF5] hover:bg-[#151D2A] transition-colors focus-visible:outline-2 focus-visible:outline-[#648CFF] cursor-pointer h-10',
                collapsed
                  ? 'w-full justify-center px-0'
                  : 'w-full justify-between px-3',
              )}
            >
              {!collapsed && <span>Collapse</span>}
              {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}

              {collapsed && (
                <div
                  role="tooltip"
                  className="absolute left-full ml-3 px-3 py-1.5 rounded-lg bg-[#171F2D] border border-[#253247] text-xs font-medium text-[#E8EDF5] shadow-xl whitespace-nowrap opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity duration-150 z-50"
                >
                  Expand sidebar
                </div>
              )}
            </button>
          </div>
        )}
      </aside>

      {/* Mobile Bottom Quick-Access Bar (below 768px) */}
      <nav
        aria-label="Mobile Navigation Quick Bar"
        className="md:hidden fixed bottom-0 left-0 right-0 h-14 bg-[#0D131E]/95 backdrop-blur-md border-t border-[#253247] flex items-center justify-around px-2 z-30 select-none"
      >
        {[
          { to: '/app', label: 'Dashboard', icon: LayoutDashboard, exact: true },
          { to: '/app/analyze', label: 'Analyze', icon: ScanEye, matchPrefix: '/app/analyses' },
          { to: '/app/history', label: 'History', icon: History },
          { to: '/app/library', label: 'Library', icon: FolderArchive },
          { to: '/app/settings', label: 'Settings', icon: Settings },
        ].map((item) => {
          const Icon = item.icon;
          const active = isItemActive(item);

          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={clsx(
                'flex flex-col items-center justify-center min-w-[50px] min-h-[44px] py-1 px-1 rounded-lg text-[10px] font-sans font-medium transition-colors',
                active
                  ? 'text-[#648CFF]'
                  : 'text-[#91A0B8] hover:text-[#E8EDF5]',
              )}
            >
              <Icon size={18} className={active ? 'text-[#648CFF]' : 'currentColor'} />
              <span className="truncate mt-0.5 leading-none">{item.label}</span>
            </NavLink>
          );
        })}
      </nav>
    </>
  );
};
