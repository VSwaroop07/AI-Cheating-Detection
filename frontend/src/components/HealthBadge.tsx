/**
 * HealthBadge component displaying backend connectivity state.
 */

import React from "react";
import { useHealth } from "../hooks/useHealth";

export const HealthBadge: React.FC = () => {
  const { health, loading, error, refetch } = useHealth();

  if (loading) {
    return (
      <span className="health-badge loading" title="Checking backend health...">
        Checking backend...
      </span>
    );
  }

  if (error) {
    return (
      <span
        className="health-badge error"
        title={`Error: ${error}. Click to retry.`}
        onClick={() => refetch()}
        style={{ cursor: "pointer" }}
      >
        Backend: Offline (Click to retry)
      </span>
    );
  }

  return (
    <span
      className="health-badge healthy"
      title={`Database: ${health?.database} | Time: ${health?.timestamp}`}
      onClick={() => refetch()}
      style={{ cursor: "pointer" }}
    >
      Backend: Online ({health?.service})
    </span>
  );
};

