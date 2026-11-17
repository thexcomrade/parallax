import React from 'react';
import { clsx } from 'clsx';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'elevated' | 'subtle' | 'interactive';
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ variant = 'default', padding = 'md', className, children, ...props }, ref) => {
    const variantStyles = {
      default:
        'bg-[var(--surface-card)] border border-[var(--border-default)] shadow-[var(--shadow-card)]',
      elevated:
        'bg-[var(--surface-elevated)] border border-[var(--border-strong)] shadow-[var(--shadow-elevated)]',
      subtle:
        'bg-[var(--surface-card)]/60 border border-[var(--border-subtle)]',
      interactive:
        'bg-[var(--surface-card)] border border-[var(--border-default)] hover:border-[var(--cobalt-500)]/60 hover:bg-[var(--surface-hover)] transition-all duration-150 cursor-pointer shadow-[var(--shadow-card)] hover:shadow-[var(--shadow-elevated)]',
    }[variant];

    const paddingStyles = {
      none: '',
      sm: 'p-3 sm:p-4',
      md: 'p-4 sm:p-5 md:p-6',
      lg: 'p-6 sm:p-8',
    }[padding];

    return (
      <div
        ref={ref}
        className={clsx(
          'relative rounded-[var(--radius-md)] overflow-hidden text-[var(--text-100)]',
          variantStyles,
          paddingStyles,
          className,
        )}
        {...props}
      >
        {children}
      </div>
    );
  },
);

Card.displayName = 'Card';

export const CardHeader: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className,
  children,
  ...props
}) => (
  <div
    className={clsx('flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4', className)}
    {...props}
  >
    {children}
  </div>
);

export interface CardTitleProps extends React.HTMLAttributes<HTMLHeadingElement> {
  as?: 'h2' | 'h3' | 'h4';
}

export const CardTitle: React.FC<CardTitleProps> = ({
  as: Component = 'h2',
  className,
  children,
  ...props
}) => (
  <Component
    className={clsx(
      'font-sans font-bold text-base md:text-lg text-[var(--text-primary)] tracking-tight',
      className,
    )}
    {...props}
  >
    {children}
  </Component>
);

export const CardDescription: React.FC<React.HTMLAttributes<HTMLParagraphElement>> = ({
  className,
  children,
  ...props
}) => (
  <p
    className={clsx('font-sans text-xs text-[var(--text-300)] leading-relaxed mt-0.5', className)}
    {...props}
  >
    {children}
  </p>
);

export const CardContent: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className,
  children,
  ...props
}) => (
  <div className={clsx('flex-1', className)} {...props}>
    {children}
  </div>
);

export const CardFooter: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className,
  children,
  ...props
}) => (
  <div
    className={clsx(
      'mt-4 pt-4 border-t border-[var(--border-subtle)] flex items-center justify-between gap-3 text-xs',
      className,
    )}
    {...props}
  >
    {children}
  </div>
);
