import { createBrowserRouter, Navigate } from 'react-router-dom';
import { LoginPage } from '@/features/auth/LoginPage';
import { RegisterPage } from '@/features/auth/RegisterPage';
import { ProtectedRoute } from '@/components/layout/ProtectedRoute';
import { AppShell } from '@/components/layout/AppShell';
import { DashboardPage } from '@/features/dashboard/DashboardPage';
import { AnalyzePage } from '@/features/analyze/AnalyzePage';
import { ReportPage } from '@/features/report/ReportPage';
import { HistoryPage } from '@/features/history/HistoryPage';

function comingSoon(title: string, code: string, phase: string) {
  return (
    <div className="max-w-4xl mx-auto flex flex-col gap-4 py-8">
      <div className="font-mono text-xs uppercase text-[var(--cobalt-400)]">
        CHANNEL // {code}
      </div>
      <h1 className="font-display font-black text-2xl uppercase tracking-tight text-white">
        {title}
      </h1>
      <div className="p-6 bg-[var(--black-900)] border border-[var(--line)] rounded-[var(--radius-md)] font-mono text-xs text-[var(--text-300)] flex flex-col gap-3">
        <div>STATUS: CHANNEL CONFIGURED FOR BUILD IN {phase}</div>
        <div className="text-[var(--text-500)]">
          Pipeline modules and user interface instrumentation will activate upon phase release.
        </div>
      </div>
    </div>
  );
}

export const router = createBrowserRouter([
  {
    path: '/login',
    element: <LoginPage />,
  },
  {
    path: '/register',
    element: <RegisterPage />,
  },
  {
    path: '/app',
    element: <ProtectedRoute />,
    children: [
      {
        element: <AppShell />,
        children: [
          {
            index: true,
            element: <DashboardPage />,
          },
          {
            path: 'analyze',
            element: <AnalyzePage />,
          },
          {
            path: 'analyses/:id',
            element: <ReportPage />,
          },
          {
            path: 'history',
            element: <HistoryPage />,
          },
          {
            path: 'library',
            element: comingSoon('Sample Corpus Library', 'CH-04', 'PHASE 8'),
          },
          {
            path: 'intel',
            element: comingSoon('Threat Radar & Intel', 'CH-05', 'PHASE 8'),
          },
          {
            path: 'model',
            element: comingSoon('Model Card & Metrics', 'CH-06', 'PHASE 9'),
          },
          {
            path: 'settings',
            element: comingSoon('Analyst Account Settings', 'CH-07', 'PHASE 9'),
          },
        ],
      },
    ],
  },
  {
    path: '/',
    element: <Navigate to="/app" replace />,
  },
  {
    path: '*',
    element: <Navigate to="/app" replace />,
  },
]);
