import React from 'react';
import { clsx } from 'clsx';

export interface TabItem {
  id: string;
  label: string;
  count?: number | string;
  icon?: React.ReactNode;
}

export interface TabsProps {
  tabs: TabItem[];
  activeId: string;
  onChange: (id: string) => void;
  className?: string;
  variant?: 'underline' | 'pills';
}

export const Tabs: React.FC<TabsProps> = ({
  tabs,
  activeId,
  onChange,
  className,
  variant = 'pills',
}) => {
  const handleKeyDown = (e: React.KeyboardEvent, index: number) => {
    let nextIndex: number | null = null;
    if (e.key === 'ArrowRight') {
      nextIndex = (index + 1) % tabs.length;
    } else if (e.key === 'ArrowLeft') {
      nextIndex = (index - 1 + tabs.length) % tabs.length;
    } else if (e.key === 'Home') {
      nextIndex = 0;
    } else if (e.key === 'End') {
      nextIndex = tabs.length - 1;
    }

    if (nextIndex !== null) {
      e.preventDefault();
      onChange(tabs[nextIndex].id);
      const buttons = (e.currentTarget.parentElement?.querySelectorAll('button[role="tab"]')) as NodeListOf<HTMLButtonElement> | undefined;
      buttons?.[nextIndex]?.focus();
    }
  };

  return (
    <div
      role="tablist"
      aria-orientation="horizontal"
      className={clsx(
        variant === 'pills'
          ? 'inline-flex items-center gap-1 p-1 bg-[var(--surface-card)] border border-[var(--border-default)] rounded-[var(--radius-md)]'
          : 'flex items-center gap-6 border-b border-[var(--border-default)] w-full',
        className,
      )}
    >
      {tabs.map((tab, idx) => {
        const isActive = tab.id === activeId;

        const pillStyles = clsx(
          'flex items-center gap-2 px-3 py-1.5 rounded-[var(--radius-sm)] text-xs font-sans font-medium transition-all select-none',
          isActive
            ? 'bg-[var(--surface-elevated)] text-[var(--text-100)] border border-[var(--border-strong)] shadow-xs font-semibold'
            : 'text-[var(--text-400)] hover:text-[var(--text-200)] hover:bg-[var(--surface-hover)] border border-transparent',
          'focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)]',
        );

        const underlineStyles = clsx(
          'relative flex items-center gap-2 py-3 text-sm font-sans transition-colors select-none -mb-px',
          isActive
            ? 'text-[var(--text-100)] font-semibold border-b-2 border-[var(--cobalt-500)]'
            : 'text-[var(--text-400)] hover:text-[var(--text-200)] border-b-2 border-transparent',
          'focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)]',
        );

        return (
          <button
            key={tab.id}
            role="tab"
            type="button"
            aria-selected={isActive}
            tabIndex={isActive ? 0 : -1}
            onClick={() => onChange(tab.id)}
            onKeyDown={(e) => handleKeyDown(e, idx)}
            className={variant === 'pills' ? pillStyles : underlineStyles}
          >
            {tab.icon && <span className="shrink-0">{tab.icon}</span>}
            <span>{tab.label}</span>
            {tab.count !== undefined && (
              <span
                className={clsx(
                  'px-1.5 py-0.2 rounded-full font-mono text-[10px]',
                  isActive
                    ? 'bg-[var(--cobalt-500)]/20 text-[var(--cobalt-300)]'
                    : 'bg-[var(--surface-hover)] text-[var(--text-400)]',
                )}
              >
                {tab.count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
};
