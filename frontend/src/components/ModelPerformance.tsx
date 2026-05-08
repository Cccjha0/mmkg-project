import { useMemo, useState } from "react";
import { Activity, Target, TrendingUp } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card } from "./ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { usePerformanceData } from "../hooks/usePerformanceData";

type ModelFilter =
  | "all"
  | "text-complex"
  | "text-rgcn"
  | "early-fusion"
  | "residual-gate";

const MODEL_LABEL_MAP: Record<ModelFilter, string> = {
  all: "All Models",
  "text-complex": "Text-ComplEx",
  "text-rgcn": "Text-RGCN",
  "early-fusion": "Early Fusion",
  "residual-gate": "Residual+Gate",
};

export function ModelPerformance() {
  const [dataset, setDataset] = useState("OpenBG-IMG");
  const [modelFilter, setModelFilter] = useState<ModelFilter>("all");
  const [metric] = useState("accuracy");

  const { overview, curves, comparison, loading, error, reload } =
    usePerformanceData({ dataset });

  const filteredComparisonRows = useMemo(() => {
    if (!comparison?.rows) return [];

    if (modelFilter === "all") {
      return comparison.rows.filter((row) => row.dataset === dataset);
    }

    const selectedModelName = MODEL_LABEL_MAP[modelFilter];
    return comparison.rows.filter(
      (row) => row.dataset === dataset && row.model === selectedModelName
    );
  }, [comparison, dataset, modelFilter]);

  const accuracyCurveData = useMemo(() => {
    if (!curves?.epochs || !curves?.series) return [];

    const allowedModels =
      modelFilter === "all" ? null : [MODEL_LABEL_MAP[modelFilter]];

    return curves.epochs.map((epoch, index) => {
      const point: Record<string, string | number | null> = { epoch };

      curves.series.forEach((series) => {
        if (!allowedModels || allowedModels.includes(series.model)) {
          point[series.model] = series.values?.[index] ?? null;
        }
      });

      return point;
    });
  }, [curves, modelFilter]);

  const visibleSeries = useMemo(() => {
    if (!curves?.series) return [];

    if (modelFilter === "all") {
      return curves.series;
    }

    const selectedModelName = MODEL_LABEL_MAP[modelFilter];
    return curves.series.filter((series) => series.model === selectedModelName);
  }, [curves, modelFilter]);

  const datasetGroupsText = useMemo(() => {
    if (!overview?.dataset_groups?.length) return "-";

    return overview.dataset_groups
      .map((group) => `${group.dataset}: ${group.models.join(", ")}`)
      .join(" | ");
  }, [overview]);

  const bestRowForDataset = useMemo(() => {
    if (!filteredComparisonRows.length) return null;

    return [...filteredComparisonRows].sort(
      (a, b) => b.accuracy - a.accuracy
    )[0];
  }, [filteredComparisonRows]);

  if (loading) {
    return (
      <div className="mx-auto max-w-[1440px] space-y-6 px-8 py-8">
        <div className="space-y-2">
          <h1>Model Performance</h1>
          <p className="text-muted-foreground">
            Loading accuracy-based experiment results for OpenBG-500 and
            OpenBG-IMG...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-[1440px] space-y-6 px-8 py-8">
        <div className="space-y-2">
          <h1>Model Performance</h1>
          <p className="text-muted-foreground">
            Accuracy-based comparison of text-only and multimodal models on
            OpenBG-500 and OpenBG-IMG
          </p>
          <p className="text-red-600">{error}</p>
        </div>

        <Card className="p-6">
          <div className="space-y-3">
            <p className="text-muted-foreground">
              Failed to load performance data from the backend.
            </p>
            <button
              onClick={() => void reload()}
              className="rounded-lg bg-primary px-4 py-2 text-primary-foreground"
            >
              Retry
            </button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1440px] space-y-6 px-8 py-8">
      <div className="space-y-2">
        <h1>Model Performance</h1>
        <p className="text-muted-foreground">
          Accuracy-based comparison of text-only and multimodal models on
          OpenBG-500 and OpenBG-IMG
        </p>
      </div>

      <Card className="p-4">
        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-3">
            <label className="mb-2 block">Model</label>
            <Select
              value={modelFilter}
              onValueChange={(value) => setModelFilter(value as ModelFilter)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Models</SelectItem>
                <SelectItem value="text-complex">Text-ComplEx</SelectItem>
                <SelectItem value="text-rgcn">Text-RGCN</SelectItem>
                <SelectItem value="early-fusion">Early Fusion</SelectItem>
                <SelectItem value="residual-gate">Residual+Gate</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="col-span-3">
            <label className="mb-2 block">Dataset</label>
            <Select value={dataset} onValueChange={setDataset}>
              <SelectTrigger>
                <SelectValue placeholder="Select dataset" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="OpenBG-500">OpenBG-500</SelectItem>
                <SelectItem value="OpenBG-IMG">OpenBG-IMG</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="col-span-3">
            <label className="mb-2 block">Metric</label>
            <Select value={metric} disabled>
              <SelectTrigger>
                <SelectValue placeholder="Select metric" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="accuracy">Accuracy</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-12 gap-6">
        <Card className="col-span-4 p-6">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <p className="text-muted-foreground">Best Accuracy Model</p>
              <p className="text-3xl">{overview?.best_accuracy_model ?? "-"}</p>
              <div className="flex items-center gap-1 text-muted-foreground">
                <Target className="h-4 w-4" />
                <span className="text-sm">
                  Current top model across uploaded results
                </span>
              </div>
            </div>
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <Target className="h-6 w-6 text-primary" />
            </div>
          </div>
        </Card>

        <Card className="col-span-4 p-6">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <p className="text-muted-foreground">Best Accuracy</p>
              <p className="text-3xl">
                {overview ? overview.best_accuracy.toFixed(3) : "-"}
              </p>
              <div className="flex items-center gap-1 text-muted-foreground">
                <TrendingUp className="h-4 w-4" />
                <span className="text-sm">
                  Primary metric for the current performance page
                </span>
              </div>
            </div>
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-chart-2/10">
              <TrendingUp className="h-6 w-6 text-chart-2" />
            </div>
          </div>
        </Card>

        <Card className="col-span-4 p-6">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <p className="text-muted-foreground">Number of Models</p>
              <p className="text-3xl">{overview?.num_models ?? 0}</p>
              <div className="flex items-center gap-1 text-muted-foreground">
                <Activity className="h-4 w-4" />
                <span className="text-sm">
                  Last updated: {overview?.last_updated ?? "-"}
                </span>
              </div>
            </div>
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-chart-3/10">
              <Activity className="h-6 w-6 text-chart-3" />
            </div>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-12 gap-6">
        <Card className="col-span-7 p-6">
          <div className="space-y-4">
            <div>
              <h3>Model Comparison</h3>
              <p className="text-sm text-muted-foreground">
                Accuracy, MRR, and Hits@10 across real project models
              </p>
            </div>

            {filteredComparisonRows.length === 0 ? (
              <div className="flex h-[300px] items-center justify-center text-sm text-muted-foreground">
                No comparison data available for the current selection.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={filteredComparisonRows}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="model" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "white",
                      border: "1px solid #e5e7eb",
                      borderRadius: "0.5rem",
                    }}
                  />
                  <Legend />
                  <Bar
                    dataKey="accuracy"
                    fill="#0891b2"
                    name="Accuracy"
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar
                    dataKey="hits10"
                    fill="#06b6d4"
                    name="Hits@10"
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar
                    dataKey="mrr"
                    fill="#6366f1"
                    name="MRR"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>

        <Card className="col-span-5 p-6">
          <div className="space-y-4">
            <div>
              <h3>Accuracy Trend Across Models</h3>
              <p className="text-sm text-muted-foreground">
                Accuracy curves by epoch for {dataset}
              </p>
            </div>

            {accuracyCurveData.length === 0 ? (
              <div className="flex h-[300px] items-center justify-center text-sm text-muted-foreground">
                No accuracy curve data available for the current selection.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={accuracyCurveData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="epoch"
                    stroke="#6b7280"
                    label={{ value: "Epoch", position: "insideBottom", offset: -5 }}
                  />
                  <YAxis stroke="#6b7280" domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "white",
                      border: "1px solid #e5e7eb",
                      borderRadius: "0.5rem",
                    }}
                  />
                  <Legend />
                  {visibleSeries.map((series, index) => {
                    const colors = ["#0891b2", "#06b6d4", "#6366f1", "#8b5cf6"];
                    return (
                      <Line
                        key={series.model}
                        type="monotone"
                        dataKey={series.model}
                        stroke={colors[index % colors.length]}
                        strokeWidth={2}
                        name={series.model}
                        dot={{ r: 3 }}
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>
      </div>

      <Card className="p-6">
        <h3 className="mb-4">Performance Summary</h3>
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-8">
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b py-2">
                <span>Current Dataset</span>
                <span>{dataset}</span>
              </div>

              <div className="flex items-center justify-between border-b py-2">
                <span>Best Model in Current View</span>
                <span>{bestRowForDataset?.model ?? "-"}</span>
              </div>

              <div className="flex items-center justify-between border-b py-2">
                <span>Best Accuracy in Current View</span>
                <span>
                  {bestRowForDataset
                    ? `${(bestRowForDataset.accuracy * 100).toFixed(1)}%`
                    : "-"}
                </span>
              </div>

              <div className="flex items-center justify-between border-b py-2">
                <span>Best Epoch</span>
                <span>{bestRowForDataset?.best_epoch ?? "-"}</span>
              </div>

              <div className="flex items-center justify-between py-2">
                <span>Available Dataset Groups</span>
                <span className="max-w-[70%] text-right text-sm text-muted-foreground">
                  {datasetGroupsText}
                </span>
              </div>
            </div>
          </div>

          <div className="col-span-4 rounded-lg bg-muted/30 p-4">
            <h4 className="mb-3">Key Insights</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>- OpenBG-500 is used for text-only model comparison.</li>
              <li>- OpenBG-IMG is used for multimodal model comparison.</li>
              <li>- Text-ComplEx and Text-RGCN provide text-based baselines.</li>
              <li>- Early Fusion and Residual+Gate represent multimodal approaches.</li>
              <li>
                - Residual+Gate shows stronger performance on OpenBG-IMG in our
                current results.
              </li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}