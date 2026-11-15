import React from 'react';
import { clsx } from 'clsx';

export interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'rect' | 'text' | 'circle';
}

export const Skeleton: React.FC<SkeletonProps> = ({
  variant = 'rect',
  className,
  ...props
}) => {
  const baseStyles =
    'relative overflow-hidden bg-[var(--black-800)]/80 animate-pulse';

  const variantStyles = {
    rect: 'rounded-[var(--radius-sm)]',
    text: 'h-4 w-full rounded-xs',
    circle: 'rounded-full',
  }[variant];

  return (
    <div
      className={clsx(baseStyles, variantStyles, className)}
      aria-hidden="true"
      {...props}
    >
      <span className="absolute inset-0 -translate-x-full animate-[shimmer_2s_infinite] bg-gradient-to-r from-transparent via-white/5 to-transparent" />
    </div>
  );
};
