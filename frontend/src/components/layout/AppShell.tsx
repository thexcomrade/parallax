import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { motion, AnimatePresence, useReducedMotion } from 'motion/react';
import { StatusStrip } from './StatusStrip';
import { CommandRail } from './CommandRail';
import { DUR, EASE_INSTRUMENT } from '@/lib/motion';

export const AppShell: React.FC = () => {
  const location = useLocation();
  const shouldReduceMotion = useReducedMotion();

  return (
    <div className="min-h-screen canvas-bg text-[var(--text-100)] flex flex-col font-sans">
      <StatusStrip />
      <div className="flex-1 flex overflow-hidden">
        <CommandRail />
        <main
          id="main-content"
          tabIndex={-1}
          className="flex-1 overflow-y-auto overflow-x-hidden p-4 md:p-6 lg:p-8 pb-20 md:pb-8 focus:outline-none"
        >
          <div className="canvas-container">
            <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={shouldReduceMotion ? { opacity: 1 } : { opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={shouldReduceMotion ? { opacity: 1 } : { opacity: 0, y: -4 }}
                transition={{ duration: DUR.route, ease: EASE_INSTRUMENT }}
              >
                <Outlet />
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  );
};
