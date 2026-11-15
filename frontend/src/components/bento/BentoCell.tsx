import React from 'react';
import { clsx } from 'clsx';

interface BentoCellProps {
  label: string;
  status?: string;
  area?: string;
  colSpan?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
  rowSpan?: 1 | 2 | 3 | 4;
  className?: string;
  bodyClassName?: string;
  hazard?: boolean;
  busy?: boolean;
  children: React.ReactNode;
}

export const BentoCell: React.FC<BentoCellProps> = ({
  label,
  status,
  area,
  colSpan,
  rowSpan,
  className,
  bodyClassName,
  hazard,
  busy,
  children,
}) => {
  const colSpanStyle = colSpan ? `span ${colSpan} / span ${colSpan}` : undefined;
  const rowSpanStyle = rowSpan ? `span ${rowSpan} / span ${rowSpan}` : undefined;

  return (
    <section
      className={clsx(
        'bento-cell',
        hazard && 'hazard-stripes',
        busy && 'opacity-60 pointer-events-none',
        className
      )}
      style={{
        gridArea: area,
        gridColumn: colSpanStyle,
        gridRow: rowSpanStyle,
      }}
      aria-label={label}
    >
      {/* Corner Crop Marks */}
      <span className="crop-mark crop-tl" aria-hidden="true" />
      <span className="crop-mark crop-tr" aria-hidden="true" />
      <span className="crop-mark crop-bl" aria-hidden="true" />
      <span className="crop-mark crop-br" aria-hidden="true" />

      {/* Laboratory Cell Header */}
      <header className="bento-cell-header">
        <span className="bento-cell-label">
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--cobalt-400)] inline-block" />
          {label}
        </span>
        {status && <span className="bento-cell-status">{status}</span>}
      </header>

      {/* Cell Body */}
      <div className={clsx('bento-cell-body', bodyClassName)}>
        {children}
      </div>
    </section>
  );
};
