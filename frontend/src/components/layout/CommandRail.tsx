import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  ScanEye,
  History,
  FolderArchive,
  Activity,
  Cpu,
  Settings,
  type LucideIcon,
} from 'lucide-react';
import { clsx } from 'clsx';

interface NavItem {
  to: string;
  label: string;
  code: string;
  icon: LucideIcon;
}

const NAV_ITEMS: NavItem[] = [
  { to: '/app', label: 'Dashboard', code: '01', icon: LayoutDashboard },
  { to: '/app/analyze', label: 'Analyze', code: '02', icon: ScanEye },
  { to: '/app/history', label: 'History', code: '03', icon: History },
  { to: '/app/library', label: 'Library', code: '04', icon: FolderArchive },
  { to: '/app/intel', label: 'Intel', code: '05', icon: Activity },
  { to: '/app/model', label: 'Model Card', code: '06', icon: Cpu },
  { to: '/app/settings', label: 'Settings', code: '07', icon: Settings },
];

export const CommandRail: React.FC = () => {
  return (
    <>
      {/* Desktop / Tablet Left Rail (md and up) */}
      <aside className="hidden md:flex flex-col w-56 lg:w-60 bg-[var(--black-900)] border-r border-[var(--line)] shrink-0 select-none py-3 px-2 z-10">
        <div className="px-3 pb-3 mb-2 border-b border-[var(--line)] flex items-center justify-between">
          <span className="font-mono text-[10px] uppercase tracking-widest text-[var(--text-500)]">
            INSTRUMENT CHANNELS
          </span>
          <span className="font-mono text-[9px] text-[var(--cobalt-400)]">v1.0</span>
        </div>

        <nav aria-label="Primary Navigation" className="flex flex-col gap-1 flex-1">
          {NAV_ITEMS.map(({ to, label, code, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/app'}
              className={({ isActive }) =>
                clsx(
                  'relative flex items-center gap-3 px-3 py-2.5 rounded-[var(--radius-sm)] text-xs font-mono font-medium transition-colors focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)]',
                  isActive
                    ? 'bg-[var(--black-800)] text-[var(--text-100)] border-l-2 border-[var(--cobalt-500)] shadow-[inset_0_0_12px_rgba(31,77,255,0.08)]'
                    : 'text-[var(--text-300)] hover:text-white hover:bg-[var(--black-800)]/60 border-l-2 border-transparent'
                )
              }
            >
              {({ isActive }) => (
                <>
                  <Icon
                    size={16}
                    className={isActive ? 'text-[var(--cobalt-400)] shrink-0' : 'text-[var(--text-500)] shrink-0'}
                  />
                  <span className="truncate flex-1 tracking-wider uppercase">{label}</span>
                  <span
                    className={clsx(
                      'text-[10px] font-mono',
                      isActive ? 'text-[var(--lime-400)]' : 'text-[var(--text-500)]'
                    )}
                  >
                    [{code}]
                  </span>
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="pt-3 border-t border-[var(--line)] px-3 text-[10px] font-mono text-[var(--text-500)] flex flex-col gap-1">
          <div>CH-F: VI-T B16</div>
          <div>CH-A: LOG-MEL D256</div>
          <div className="text-[var(--lime-400)]">FUSION: DWF v1</div>
        </div>
      </aside>

      {/* Mobile Bottom Tab Bar (under 768px) */}
      <nav
        aria-label="Mobile Navigation"
        className="md:hidden fixed bottom-0 left-0 right-0 h-16 bg-[var(--black-900)] border-t border-[var(--line)] flex items-center justify-around px-1 z-30 select-none"
      >
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/app'}
            className={({ isActive }) =>
              clsx(
                'flex flex-col items-center justify-center min-w-[44px] min-h-[44px] py-1 px-1 rounded text-[10px] font-mono tracking-tight transition-colors',
                isActive
                  ? 'text-[var(--lime-400)] border-t-2 border-[var(--lime-400)]'
                  : 'text-[var(--text-500)] hover:text-[var(--text-300)]'
              )
            }
          >
            {({ isActive }) => (
              <>
                <Icon size={18} className={isActive ? 'text-[var(--lime-400)]' : 'currentColor'} />
                <span className="truncate max-w-[48px] scale-90">{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>
    </>
  );
};
