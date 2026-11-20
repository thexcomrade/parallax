import React, { Suspense } from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import { ProtectedRoute } from '@/components/layout/ProtectedRoute';
import { AppShell } from '@/components/layout/AppShell';
import { RouteLoadingFallback } from '@/components/layout/RouteLoadingFallback';

const LoginPage = React.lazy(() =>
  import('@/features/auth/LoginPage').then((m) => ({ default: m.LoginPage }))
);
const RegisterPage = React.lazy(() =>
  import('@/features/auth/RegisterPage').then((m) => ({ default: m.RegisterPage }))
);
const DashboardPage = React.lazy(() =>
  import('@/features/dashboard/DashboardPage').then((m) => ({ default: m.DashboardPage }))
);
const AnalyzePage = React.lazy(() =>
  import('@/features/analyze/AnalyzePage').then((m) => ({ default: m.AnalyzePage }))
);
const ReportPage = React.lazy(() =>
  import('@/features/report/ReportPage').then((m) => ({ default: m.ReportPage }))
);
const HistoryPage = React.lazy(() =>
  import('@/features/history/HistoryPage').then((m) => ({ default: m.HistoryPage }))
);
const LibraryPage = React.lazy(() =>
  import('@/features/library/LibraryPage').then((m) => ({ default: m.LibraryPage }))
);
const IntelPage = React.lazy(() =>
  import('@/features/intel/IntelPage').then((m) => ({ default: m.IntelPage }))
);
const SettingsPage = React.lazy(() =>
  import('@/features/settings/SettingsPage').then((m) => ({ default: m.SettingsPage }))
);

const lazyElement = (Component: React.ComponentType) => (
  <Suspense fallback={<RouteLoadingFallback />}>
    <Component />
  </Suspense>
);

export const router = createBrowserRouter([
  {
    path: '/login',
    element: lazyElement(LoginPage),
  },
  {
    path: '/register',
    element: lazyElement(RegisterPage),
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
            element: lazyElement(DashboardPage),
          },
          {
            path: 'dashboard',
            element: <Navigate to="/app" replace />,
          },
          {
            path: 'analyze',
            element: lazyElement(AnalyzePage),
          },
          {
            path: 'analyses/:id',
            element: lazyElement(ReportPage),
          },
          {
            path: 'history',
            element: lazyElement(HistoryPage),
          },
          {
            path: 'library',
            element: lazyElement(LibraryPage),
          },
          {
            path: 'intel',
            element: lazyElement(IntelPage),
          },
          {
            path: 'model',
            element: <Navigate to="/app" replace />,
          },
          {
            path: 'settings',
            element: lazyElement(SettingsPage),
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
