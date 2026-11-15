import { useEffect, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { authStore } from '@/lib/auth-store';
import { api, apiFetch } from '@/lib/api';
import type { TokenResponse, UserResponse } from '@/types/api';
import type { LoginFormData, RegisterFormData } from '@/features/auth/authSchemas';

let initialRefreshPromise: Promise<boolean> | null = null;

export function useAuth() {
  const queryClient = useQueryClient();
  const [user, setUser] = useState<UserResponse | null>(() => authStore.getUser());
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => authStore.isAuthenticated());
  const [isInitializing, setIsInitializing] = useState<boolean>(!authStore.isAuthenticated());

  useEffect(() => {
    const unsubscribe = authStore.subscribe(() => {
      setUser(authStore.getUser());
      setIsAuthenticated(authStore.isAuthenticated());
    });

    if (!authStore.isAuthenticated()) {
      if (!initialRefreshPromise) {
        initialRefreshPromise = (async () => {
          try {
            const data = await apiFetch<TokenResponse>('/auth/refresh', {
              method: 'POST',
              _retry: true,
            });
            authStore.set(data.access_token, data.user);
            return true;
          } catch {
            authStore.clear();
            return false;
          } finally {
            setIsInitializing(false);
          }
        })();
      } else {
        initialRefreshPromise.finally(() => {
          setIsInitializing(false);
        });
      }
    } else {
      setIsInitializing(false);
    }

    return () => {
      unsubscribe();
    };
  }, []);

  const loginMutation = useMutation({
    mutationFn: async (credentials: LoginFormData) => {
      const data = await api.post<TokenResponse>('/auth/login', credentials);
      authStore.set(data.access_token, data.user);
      return data;
    },
    onSuccess: () => {
      queryClient.clear();
    },
  });

  const registerMutation = useMutation({
    mutationFn: async (payload: RegisterFormData) => {
      const data = await api.post<UserResponse>('/auth/register', {
        email: payload.email,
        password: payload.password,
      });
      return data;
    },
  });

  const logoutMutation = useMutation({
    mutationFn: async () => {
      try {
        await api.post('/auth/logout');
      } finally {
        authStore.clear();
        queryClient.clear();
      }
    },
  });

  return {
    user,
    isAuthenticated,
    isInitializing,
    login: loginMutation.mutateAsync,
    isLoggingIn: loginMutation.isPending,
    loginError: loginMutation.error,
    register: registerMutation.mutateAsync,
    isRegistering: registerMutation.isPending,
    registerError: registerMutation.error,
    logout: logoutMutation.mutateAsync,
    isLoggingOut: logoutMutation.isPending,
  };
}
