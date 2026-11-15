// In-memory access token store.
// NEVER persisted to localStorage or sessionStorage.
// Token lives only in this module's closure for the page lifetime.

import type { UserResponse } from '@/types/api';

interface AuthState {
  accessToken: string | null;
  user: UserResponse | null;
}

let state: AuthState = {
  accessToken: null,
  user: null,
};

type Listener = () => void;
const listeners = new Set<Listener>();

function notify(): void {
  listeners.forEach((fn) => fn());
}

export const authStore = {
  getToken(): string | null {
    return state.accessToken;
  },

  getUser(): UserResponse | null {
    return state.user;
  },

  isAuthenticated(): boolean {
    return state.accessToken !== null;
  },

  set(accessToken: string, user: UserResponse): void {
    state = { accessToken, user };
    notify();
  },

  clear(): void {
    state = { accessToken: null, user: null };
    notify();
  },

  subscribe(listener: Listener): () => void {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },
} as const;
