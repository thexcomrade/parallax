import React from 'react';
import { clsx } from 'clsx';

export interface PageHeaderProps {
  title: string;
  description?: string;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  breadcrumbs?: Array<{ label: string; href?: string }>;
  className?: string;
}

export const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  description,
  badge,
  actions,
  breadcrumbs,
  className,
}) => {
  return (
    <header
      className={clsx(
        'flex flex-col md:flex-row md:items-end justify-between gap-4 pb-5 mb-6 border-b border-[var(--border-subtle)]',
        className,
      )}
    >
      <div className="flex flex-col gap-1.5 max-w-3xl min-w-0 flex-1">
        {breadcrumbs && breadcrumbs.length > 0 && (
          <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 text-xs text-[var(--text-400)] mb-1">
            {breadcrumbs.map((crumb, idx) => (
              <React.Fragment key={crumb.label}>
                {idx > 0 && <span className="text-[var(--text-500)] select-none">/</span>}
                {crumb.href ? (
                  <a
                    href={crumb.href}
                    className="hover:text-[var(--text-100)] transition-colors focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)] rounded"
                  >
                    {crumb.label}
                  </a>
                ) : (
                  <span className="text-[var(--text-200)] font-medium">{crumb.label}</span>
                )}
              </React.Fragment>
            ))}
          </nav>
        )}

        <div className="flex flex-wrap items-center gap-3 min-w-0">
          <h1 className="font-sans font-bold text-2xl md:text-3xl text-[var(--text-primary)] tracking-tight break-words min-w-0">
            {title}
          </h1>
          {badge && <div className="inline-flex shrink-0">{badge}</div>}
        </div>

        {description && (
          <p className="font-sans text-xs md:text-sm text-[var(--text-300)] leading-relaxed">
            {description}
          </p>
        )}
      </div>

      {actions && (
        <div className="flex flex-wrap items-center gap-2.5 shrink-0 self-start md:self-end">
          {actions}
        </div>
      )}
    </header>
  );
};
