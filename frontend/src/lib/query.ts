import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Keep data fresh for 30 s by default; individual queries override this.
      staleTime: 30_000,
      // Never automatically retry on 4xx errors (client errors, not transient).
      retry: (failureCount, error) => {
        if (error instanceof Error && 'status' in error) {
          const status = (error as Error & { status: number }).status;
          if (status >= 400 && status < 500) return false;
        }
        return failureCount < 2;
      },
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: false,
    },
  },
});
