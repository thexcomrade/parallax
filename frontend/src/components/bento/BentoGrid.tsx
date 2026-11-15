import React from 'react';
import { clsx } from 'clsx';

interface BentoGridProps {
  children: React.ReactNode;
  className?: string;
  areas?: string;
}

export const BentoGrid: React.FC<BentoGridProps> = ({ children, className, areas }) => {
  return (
    <div
      className={clsx('bento-grid', className)}
      style={areas ? { gridTemplateAreas: areas } : undefined}
    >
      {children}
    </div>
  );
};
