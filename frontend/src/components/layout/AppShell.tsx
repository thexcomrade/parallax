import React, { useState, useEffect } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { motion, AnimatePresence, useReducedMotion } from 'motion/react';
import { StatusStrip } from './StatusStrip';
import { CommandRail } from './CommandRail';
import { DUR, EASE_INSTRUMENT } from '@/lib/motion';

export const AppShell: React.FC = () => {
  const location = useLocation();
  const shouldReduceMotion = useReducedMotion();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  useEffect(() => {
    let robotsMeta = document.querySelector('meta[name="robots"]') as HTMLMetaElement | null;
    if (!robotsMeta) {
      robotsMeta = document.createElement('meta');
      robotsMeta.name = 'robots';
      document.head.appendChild(robotsMeta);
    }
    const prevRobots = robotsMeta.content;
    robotsMeta.content = 'noindex, nofollow';

    return () => {
      if (robotsMeta) {
        robotsMeta.content = prevRobots || 'index, follow';
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-[var(--surface-base)] text-[var(--text-100)] flex font-sans selection:bg-[var(--cobalt-500)]/30 selection:text-white">
      {/* Accessible Skip to Content Link */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 focus:px-4 focus:py-2 focus:bg-[var(--surface-elevated)] focus:border-2 focus:border-[var(--cobalt-400)] focus:text-[var(--accent-primary)] focus:font-mono focus:text-xs focus:rounded-[var(--radius-sm)] focus:shadow-xl focus:outline-none"
      >
        Skip to main content
      </a>

      {/* Navigation Rail / Sidebar (Desktop full-height sticky + Mobile drawer) */}
      <CommandRail
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed(!collapsed)}
        mobileOpen={mobileNavOpen}
        onCloseMobile={() => setMobileNavOpen(false)}
      />

      {/* Main Workspace Column (Header + Page Content Sharing Identical Offset) */}
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden pb-[60px] md:pb-0">
        {/* Global Header (Height: 76px, matching Sidebar Brand Header horizon) */}
        <StatusStrip
          onToggleMobileNav={() => setMobileNavOpen(!mobileNavOpen)}
          isMobileNavOpen={mobileNavOpen}
          collapsed={collapsed}
        />

        {/* Main Content Area */}
        <main
          id="main-content"
          tabIndex={0}
          className="flex-1 overflow-y-auto overflow-x-hidden px-4 md:px-8 pt-7 pb-24 md:pb-8 scroll-pb-28 md:scroll-pb-8 focus:outline-none"
        >
          <div className="w-full max-w-[1600px] mx-auto">
            <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={shouldReduceMotion ? { opacity: 1 } : { opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={shouldReduceMotion ? { opacity: 1 } : { opacity: 0, y: -4 }}
                transition={{ duration: DUR.route, ease: EASE_INSTRUMENT }}
                className="w-full"
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
