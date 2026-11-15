import React from 'react';
import { clsx } from 'clsx';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'alert';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  loading = false,
  icon,
  children,
  className,
  disabled,
  ...props
}) => {
  const baseStyles =
    'relative inline-flex items-center justify-center font-mono font-bold tracking-wider uppercase transition-all duration-150 select-none cursor-pointer focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)] focus-visible:outline-offset-2 disabled:cursor-not-allowed disabled:opacity-50';

  const sizeStyles = {
    sm: 'min-h-[44px] sm:min-h-[36px] px-3 text-xs gap-1.5 rounded-[var(--radius-sm)]',
    md: 'min-h-[44px] px-4 text-xs gap-2 rounded-[var(--radius-sm)]',
    lg: 'min-h-[48px] px-6 text-sm gap-2.5 rounded-[var(--radius-md)]',
  }[size];

  const variantStyles = {
    primary:
      'bg-[var(--lime-400)] text-[var(--lime-ink)] hover:bg-[var(--lime-300)] active:translate-y-[1px] border border-[var(--lime-400)]',
    secondary:
      'bg-[var(--black-800)] text-[var(--text-100)] border border-[var(--line-strong)] hover:border-[var(--cobalt-400)] hover:text-white active:bg-[var(--black-700)]',
    ghost:
      'bg-transparent text-[var(--text-300)] hover:bg-[var(--black-800)] hover:text-[var(--text-100)] border border-transparent',
    alert:
      'bg-[var(--alert-500)] text-[var(--black-950)] font-black hover:brightness-110 active:translate-y-[1px] border border-[var(--alert-500)]',
  }[variant];

  return (
    <button
      className={clsx(baseStyles, sizeStyles, variantStyles, className)}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <span
          className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin"
          aria-hidden="true"
        />
      ) : (
        icon && <span className="inline-flex shrink-0">{icon}</span>
      )}
      <span>{children}</span>
    </button>
  );
};
