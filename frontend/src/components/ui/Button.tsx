import React from 'react';
import { clsx } from 'clsx';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'alert' | 'outline' | 'subtle';
export type ButtonSize = 'xs' | 'sm' | 'md' | 'lg';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  loading = false,
  icon,
  iconPosition = 'left',
  children,
  className,
  disabled,
  ...props
}) => {
  const baseStyles =
    'relative inline-flex items-center justify-center font-sans font-semibold tracking-wide transition-all duration-150 select-none cursor-pointer focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)] focus-visible:outline-offset-2 disabled:cursor-not-allowed disabled:opacity-40';

  const sizeStyles = {
    xs: 'min-h-[32px] px-2.5 text-xs gap-1 rounded-[var(--radius-sm)]',
    sm: 'min-h-[36px] px-3.5 text-xs gap-1.5 rounded-[var(--radius-sm)]',
    md: 'min-h-[42px] px-4 text-sm gap-2 rounded-[var(--radius-md)]',
    lg: 'min-h-[48px] px-6 text-sm gap-2.5 rounded-[var(--radius-md)]',
  }[size];

  const variantStyles = {
    primary:
      'bg-[var(--lime-400)] text-[var(--lime-ink)] font-bold hover:bg-[var(--lime-300)] active:translate-y-[1px] border border-[var(--lime-400)] shadow-[0_2px_12px_rgba(200,255,46,0.2)]',
    secondary:
      'bg-[var(--surface-elevated)] text-[var(--text-100)] border border-[var(--border-strong)] hover:border-[var(--cobalt-400)] hover:text-white hover:bg-[var(--surface-hover)] active:bg-[var(--black-700)] shadow-[0_2px_8px_rgba(0,0,0,0.3)]',
    outline:
      'bg-transparent text-[var(--text-200)] border border-[var(--border-default)] hover:border-[var(--border-strong)] hover:text-white hover:bg-[var(--surface-card)]',
    subtle:
      'bg-[var(--surface-card)] text-[var(--text-300)] border border-[var(--border-subtle)] hover:border-[var(--border-default)] hover:text-[var(--text-100)]',
    ghost:
      'bg-transparent text-[var(--text-300)] hover:bg-[var(--surface-hover)] hover:text-[var(--text-100)] border border-transparent',
    alert:
      'bg-[var(--alert-500)] text-[var(--black-950)] font-black hover:bg-[var(--alert-400)] active:translate-y-[1px] border border-[var(--alert-500)] shadow-[0_2px_12px_rgba(255,77,94,0.25)]',
  }[variant];

  return (
    <button
      className={clsx(baseStyles, sizeStyles, variantStyles, className)}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <span
          className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin shrink-0"
          aria-hidden="true"
        />
      ) : (
        icon && iconPosition === 'left' && <span className="inline-flex shrink-0">{icon}</span>
      )}
      {children && <span>{children}</span>}
      {!loading && icon && iconPosition === 'right' && (
        <span className="inline-flex shrink-0">{icon}</span>
      )}
    </button>
  );
};
