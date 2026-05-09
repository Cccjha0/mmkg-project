import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ModelPerformance } from "../components/ModelPerformance";

vi.mock("../hooks/usePerformanceData", () => ({
  usePerformanceData: vi.fn(() => ({
    overview: {
      best_accuracy_model: "Residual+Gate",
      best_accuracy: 0.78,
      num_models: 2,
      last_updated: "2026-05-10",
      dataset_groups: [{ dataset: "OpenBG-IMG", models: ["Residual+Gate"] }],
    },
    curves: {
      metric: "accuracy",
      dataset: "OpenBG-IMG",
      epochs: [1, 2],
      series: [{ model: "Residual+Gate", values: [0.5, 0.78] }],
    },
    comparison: {
      metric: "accuracy",
      rows: [
        {
          model: "Residual+Gate",
          dataset: "OpenBG-IMG",
          accuracy: 0.78,
          mrr: 0.5,
          hits1: 0.35,
          hits3: 0.58,
          hits10: 0.78,
          best_epoch: 16,
        },
      ],
    },
    loading: false,
    error: null,
    reload: vi.fn(),
  })),
}));

describe("ModelPerformance", () => {
  it("renders overview and comparison data", () => {
    render(<ModelPerformance />);

    expect(screen.getByRole("heading", { name: "Model Performance" })).toBeInTheDocument();
    expect(screen.getAllByText("Residual+Gate").length).toBeGreaterThan(0);
    expect(screen.getByText("Performance Summary")).toBeInTheDocument();
    expect(screen.getAllByText("OpenBG-IMG").length).toBeGreaterThan(0);
  });
});
