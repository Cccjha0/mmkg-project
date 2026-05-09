import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AttributeCompletion } from "../components/AttributeCompletion";

const updatePredictedSelection = vi.fn();

vi.mock("../hooks/useAttributeCompletion", () => ({
  useAttributeCompletion: vi.fn(() => ({
    entity: {
      entity: "ent_000001",
      entity_text: "裤子",
      entity_text_zh: "裤子",
      entity_text_en: "Pants",
      has_image: false,
      image_path: null,
    },
    rows: [
      {
        relation: "rel_0001",
        relationText: "颜色",
        relationTextZh: "颜色",
        relationTextEn: "color",
        source: "Existing",
        selectedEntity: "ent_000002",
        selectedValue: "黑色",
        selectedValueZh: "黑色",
        selectedValueEn: "Black",
        selectedScore: null,
        candidates: [],
      },
      {
        relation: "rel_0002",
        relationText: "材质",
        relationTextZh: "材质",
        relationTextEn: "material",
        source: "Predicted",
        selectedEntity: "ent_000003",
        selectedValue: "棉",
        selectedValueZh: "棉",
        selectedValueEn: "Cotton",
        selectedScore: 0.72,
        candidates: [
          { entity: "ent_000003", value: "棉", valueZh: "棉", valueEn: "Cotton", score: 0.72 },
          { entity: "ent_000004", value: "麻", valueZh: "麻", valueEn: "Linen", score: 0.28 },
        ],
      },
    ],
    loading: false,
    error: null,
    reload: vi.fn(),
    updatePredictedSelection,
  })),
}));

describe("AttributeCompletion", () => {
  it("renders no-image entity state and attribute rows", () => {
    render(<AttributeCompletion />);

    expect(screen.getByRole("heading", { name: "Attribute Completion" })).toBeInTheDocument();
    expect(screen.getByText("No Image Available")).toBeInTheDocument();
    expect(screen.getByText("ent_000001")).toBeInTheDocument();
    expect(screen.getByText("Pants")).toBeInTheDocument();
    expect(screen.getByText("颜色")).toBeInTheDocument();
    expect(screen.getByText("Existing")).toBeInTheDocument();
    expect(screen.getByText("72.0%")).toBeInTheDocument();
  });

  it("keeps predicted candidate switching interactive", async () => {
    const user = userEvent.setup();
    render(<AttributeCompletion />);

    await user.selectOptions(screen.getByRole("combobox"), "ent_000004");

    await waitFor(() => {
      expect(updatePredictedSelection).toHaveBeenCalledWith("rel_0002", "ent_000004");
    });
  });
});
