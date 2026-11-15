import React from 'react';
import { Outlet } from 'react-router-dom';
import { StatusStrip } from './StatusStrip';
import { CommandRail } from './CommandRail';

export const AppShell: React.FC = () => {
  return (
    <div className="min-h-screen bg-[var(--black-950)] text-[var(--text-100)] flex flex-col font-sans">
      <StatusStrip />
      <div className="flex-1 flex overflow-hidden">
        <CommandRail />
        <main
          id="main-content"
          tabIndex={-1}
          className="flex-1 overflow-y-auto overflow-x-hidden p-4 md:p-6 pb-20 md:pb-6 focus:outline-none"
        >
          <Outlet />
        </main>
      </div>
    </div>
  );
};
