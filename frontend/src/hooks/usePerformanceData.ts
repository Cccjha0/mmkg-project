import { useCallback, useEffect, useState } from "react";
import {
  getAccuracyCurves,
  getModelComparison,
  getPerformanceOverview,
} from "../lib/api";
import type {
  AccuracyCurvesResponse,
  ModelComparisonResponse,
  PerformanceOverview,
} from "../types/api";

interface UsePerformanceDataOptions {
  dataset?: string;
}

interface UsePerformanceDataReturn {
  overview: PerformanceOverview | null;
  curves: AccuracyCurvesResponse | null;
  comparison: ModelComparisonResponse | null;
  loading: boolean;
  error: string | null;
  reload: () => Promise<void>;
}

export function usePerformanceData(
  options: UsePerformanceDataOptions = {}
): UsePerformanceDataReturn {
  const { dataset = "OpenBG-IMG" } = options;

  const [overview, setOverview] = useState<PerformanceOverview | null>(null);
  const [curves, setCurves] = useState<AccuracyCurvesResponse | null>(null);
  const [comparison, setComparison] = useState<ModelComparisonResponse | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const [overviewData, curvesData, comparisonData] = await Promise.all([
        getPerformanceOverview(),
        getAccuracyCurves(dataset),
        getModelComparison(dataset),
      ]);

      setOverview(overviewData);
      setCurves(curvesData);
      setComparison(comparisonData);
    } catch (err) {
      console.error("Failed to load performance data:", err);
      setError("Failed to load performance data.");
    } finally {
      setLoading(false);
    }
  }, [dataset]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  return {
    overview,
    curves,
    comparison,
    loading,
    error,
    reload: loadData,
  };
}