import React from 'react';
import { clsx } from 'clsx';

export type BadgeVariant =
  | 'authentic'
  | 'fake'
  | 'inconclusive'
  | 'neutral'
  | 'cobalt';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  size?: 'sm' | 'md';
  hazard?: boolean;
}

export const Badge: React.FC<BadgeProps> = ({
  variant = 'neutral',
  size = 'md',
  hazard = false,
  className,
  children,
  ...props
}) => {
  const baseStyles =
    'inline-flex items-center justify-center font-mono font-bold uppercase tracking-wider rounded-xs select-none';

  const sizeStyles = {
    sm: 'px-1.5 py-0.5 text-[10px] leading-tight',
    md: 'px-2.5 py-1 text-xs leading-none',
  }[size];

  const variantStyles = {
    authentic: 'bg-[var(--lime-400)] text-[var(--lime-ink)]',
    fake: 'bg-[var(--alert-500)] text-[var(--black-950)] font-black',
    inconclusive: 'bg-[var(--black-800)] text-[var(--text-300)] border border-[var(--line-strong)]',
    neutral: 'bg-[var(--black-800)] text-[var(--text-400)] border border-[var(--line)]',
    cobalt: 'bg-[var(--cobalt-500)] text-white',
  }[variant];

  return (
    <span
      className={clsx(
        baseStyles,
        sizeStyles,
        variantStyles,
        hazard && variant === 'fake' && 'hazard-stripes',
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
};
