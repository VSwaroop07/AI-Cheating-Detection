/**
 * Custom React hook for monitoring backend health status.
 */

import { useState, useEffect, useCallback } from "react";
import { getHealthStatus, HealthResponse } from "../services/api";

export interface UseHealthReturn {
  health: HealthResponse | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useHealth(): UseHealthReturn {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHealth = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getHealthStatus();
      setHealth(data);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to connect to backend server");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
  }, [fetchHealth]);

  return { health, loading, error, refetch: fetchHealth };
}

