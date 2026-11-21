import React from 'react';
import { clsx } from 'clsx';

export type BadgeVariant =
  | 'authentic'
  | 'fake'
  | 'inconclusive'
  | 'neutral'
  | 'cobalt'
  | 'cyan'
  | 'warning';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  size?: 'xs' | 'sm' | 'md';
  hazard?: boolean;
  dot?: boolean;
}

export const Badge: React.FC<BadgeProps> = ({
  variant = 'neutral',
  size = 'sm',
  hazard = false,
  dot = false,
  className,
  children,
  ...props
}) => {
  const baseStyles =
    'inline-flex items-center justify-center font-mono font-bold uppercase tracking-wider rounded-[var(--radius-xs)] select-none whitespace-nowrap';

  const sizeStyles = {
    xs: 'px-1.5 py-0.5 text-[9px] gap-1 leading-tight',
    sm: 'px-2 py-0.5 text-[10px] gap-1.5 leading-tight',
    md: 'px-2.5 py-1 text-xs gap-1.5 leading-none',
  }[size];

  const variantStyles = {
    authentic:
      'bg-[var(--success-500)]/15 text-[var(--success-500)] border border-[var(--success-500)]/30',
    fake:
      'bg-[var(--alert-500)]/15 text-[var(--alert-500)] border border-[var(--alert-500)]/35 font-black',
    inconclusive:
      'bg-[var(--black-800)] text-[var(--text-300)] border border-[var(--border-strong)]',
    neutral:
      'bg-[var(--surface-elevated)] text-[var(--text-400)] border border-[var(--border-default)]',
    cobalt:
      'bg-[var(--cobalt-500)]/15 text-[var(--cobalt-300)] border border-[var(--cobalt-500)]/30',
    cyan:
      'bg-[var(--cyan)]/15 text-[var(--cyan)] border border-[var(--cyan)]/30',
    warning:
      'bg-[var(--warning-500)]/15 text-[var(--warning-500)] border border-[var(--warning-500)]/30',
  }[variant];

  const dotColor = {
    authentic: 'bg-[var(--success-500)]',
    fake: 'bg-[var(--alert-500)]',
    inconclusive: 'bg-[var(--warning-500)]',
    neutral: 'bg-[var(--text-400)]',
    cobalt: 'bg-[var(--cobalt-400)]',
    cyan: 'bg-[var(--cyan)]',
    warning: 'bg-[var(--warning-500)]',
  }[variant];

  return (
    <span
      className={clsx(
        baseStyles,
        sizeStyles,
        variantStyles,
        hazard && variant === 'fake' && 'hazard-stripes',
        className,
      )}
      {...props}
    >
      {dot && (
        <span
          className={clsx('w-1.5 h-1.5 rounded-full shrink-0', dotColor)}
          aria-hidden="true"
        />
      )}
      <span>{children}</span>
    </span>
  );
};
