import { createBrowserRouter, Navigate } from 'react-router-dom';
import { LoginPage } from '@/features/auth/LoginPage';
import { RegisterPage } from '@/features/auth/RegisterPage';
import { ProtectedRoute } from '@/components/layout/ProtectedRoute';
import { AppShell } from '@/components/layout/AppShell';
import { DashboardPage } from '@/features/dashboard/DashboardPage';
import { AnalyzePage } from '@/features/analyze/AnalyzePage';
import { ReportPage } from '@/features/report/ReportPage';
import { HistoryPage } from '@/features/history/HistoryPage';
import { LibraryPage } from '@/features/library/LibraryPage';
import { IntelPage } from '@/features/intel/IntelPage';
import { SettingsPage } from '@/features/settings/SettingsPage';

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
            path: 'dashboard',
            element: <Navigate to="/app" replace />,
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
            element: <LibraryPage />,
          },
          {
            path: 'intel',
            element: <IntelPage />,
          },
          {
            path: 'model',
            element: <Navigate to="/app" replace />,
          },
          {
            path: 'settings',
            element: <SettingsPage />,
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
