import { useMemo, useState } from "react";
import {
  Sparkles,
  Package,
  ImageIcon,
  Shuffle,
  ChevronRight,
} from "lucide-react";

import { Card } from "./ui/card";
import { Badge } from "./ui/badge";
import { ImageWithFallback } from "./figma/ImageWithFallback";
import { useAttributeCompletion } from "../hooks/useAttributeCompletion";
import { demoProducts } from "../data/demoProducts";
import { resolveAssetUrl } from "../lib/api";

export function AttributeCompletion() {
  const [currentIndex, setCurrentIndex] = useState(0);

  const entityId = demoProducts[currentIndex] ?? "ent_007314";

  const { entity, rows, loading, error, updatePredictedSelection } =
    useAttributeCompletion(entityId);

  const displayText = useMemo(() => {
    return (
      entity?.entity_text ||
      entity?.entity_text_zh ||
      entity?.entity_text_en ||
      "No entity text available"
    );
  }, [entity]);

  const displayTextEn = useMemo(() => {
    return entity?.entity_text_en ?? "";
  }, [entity]);

  const imageUrl = useMemo(() => {
    return resolveAssetUrl(entity?.image_path);
  }, [entity?.image_path]);

  const handleNextProduct = () => {
    if (demoProducts.length === 0) return;
    setCurrentIndex((prev) => (prev + 1) % demoProducts.length);
  };

  const handleRandomProduct = () => {
    if (demoProducts.length <= 1) return;

    let nextIndex = currentIndex;

    while (nextIndex === currentIndex) {
      nextIndex = Math.floor(Math.random() * demoProducts.length);
    }

    setCurrentIndex(nextIndex);
  };

  const formatScore = (score: number | null) => {
    if (score === null) return "-";
    return `${(score * 100).toFixed(score >= 0.1 ? 1 : 2)}%`;
  };

  if (loading) {
    return (
      <div className="max-w-[1440px] mx-auto px-8 py-8 space-y-6">
        <div className="space-y-2">
          <h1>Attribute Completion</h1>
          <p className="text-muted-foreground">
            Residual+Gate-based entity browsing for OpenBG-IMG
          </p>
          <p className="text-muted-foreground">Loading product data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-[1440px] mx-auto px-8 py-8 space-y-6">
        <div className="space-y-2">
          <h1>Attribute Completion</h1>
          <p className="text-muted-foreground">
            Residual+Gate-based entity browsing for OpenBG-IMG
          </p>
          <p className="text-red-600">Failed to load entity data.</p>
        </div>

        <Card className="p-6">
          <p className="text-muted-foreground">
            Please try another product or retry after the backend service is
            available.
          </p>
        </Card>
      </div>
    );
  }

  return (
    <div className="max-w-[1440px] mx-auto px-8 py-8 space-y-6">
      <div className="space-y-2">
        <h1>Attribute Completion</h1>
        <p className="text-muted-foreground">
          Residual+Gate-based entity browsing for OpenBG-IMG
        </p>
      </div>

      <div className="grid grid-cols-12 gap-8">
        <div className="col-span-5">
          <Card className="p-6 space-y-5">
            <div className="flex items-center justify-between">
              <h3>Entity Information</h3>
              <Badge className="bg-green-100 text-green-800 hover:bg-green-100">
                <Sparkles className="w-3 h-3 mr-1" />
                Residual+Gate
              </Badge>
            </div>

            <div className="aspect-square bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg overflow-hidden flex items-center justify-center">
              {entity?.has_image && imageUrl ? (
                <ImageWithFallback
                  src={imageUrl}
                  alt={displayText}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center text-muted-foreground">
                  <ImageIcon className="w-10 h-10 mb-3 opacity-60" />
                  <p className="text-sm">No Image Available</p>
                </div>
              )}
            </div>

            <div className="space-y-4 pt-1">
              <div>
                <p className="text-sm text-muted-foreground">Entity ID</p>
                <div className="mt-1 flex items-center gap-2">
                  <Package className="w-4 h-4 text-muted-foreground" />
                  <p>{entity?.entity ?? entityId}</p>
                </div>
              </div>

              <div>
                <p className="text-sm text-muted-foreground">Entity Text</p>
                <div className="mt-1 space-y-1">
                  <p className="leading-6">{displayText}</p>
                  {displayTextEn ? (
                    <p className="text-sm text-muted-foreground leading-5">
                      {displayTextEn}
                    </p>
                  ) : null}
                </div>
              </div>

              <div>
                <p className="text-sm text-muted-foreground">Model</p>
                <p className="mt-1">Residual+Gate</p>
              </div>

              <div>
                <p className="text-sm text-muted-foreground">
                  Image Availability
                </p>
                <p className="mt-1">
                  {entity?.has_image ? "Available" : "Not Available"}
                </p>
              </div>
            </div>
          </Card>
        </div>

        <div className="col-span-7">
          <Card className="p-6 space-y-5">
            <div>
              <h3>Attribute Table</h3>
              <p className="text-sm text-muted-foreground mt-1">
                Existing attributes are displayed as read-only values, while
                predicted attributes support candidate switching with score
                updates.
              </p>
            </div>

            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full border-collapse">
                <thead className="bg-muted/30">
                  <tr>
                    <th className="text-left p-3 text-sm font-medium border-b">
                      Attribute Relation
                    </th>
                    <th className="text-left p-3 text-sm font-medium border-b">
                      Value
                    </th>
                    <th className="text-left p-3 text-sm font-medium border-b">
                      Source
                    </th>
                    <th className="text-left p-3 text-sm font-medium border-b">
                      Score
                    </th>
                  </tr>
                </thead>

                <tbody>
                  {rows.length === 0 ? (
                    <tr>
                      <td
                        colSpan={4}
                        className="p-6 text-center text-sm text-muted-foreground"
                      >
                        No attributes available.
                      </td>
                    </tr>
                  ) : (
                    rows.map((row) => (
                      <tr
                        key={row.relation}
                        className="border-b last:border-b-0"
                      >
                        <td className="p-3 align-middle">
                          <div className="space-y-1">
                            <p className="font-medium">{row.relationText}</p>
                            {row.relationTextEn ? (
                              <p className="text-xs text-muted-foreground">
                                {row.relationTextEn}
                              </p>
                            ) : null}
                            <p className="text-xs text-muted-foreground">
                              {row.relation}
                            </p>
                          </div>
                        </td>

                        <td className="p-3 align-middle">
                          {row.source === "Predicted" ? (
                            <select
                              value={row.selectedEntity ?? row.selectedValue}
                              onChange={(e) =>
                                updatePredictedSelection(
                                  row.relation,
                                  e.target.value
                                )
                              }
                              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                            >
                              {row.candidates.map((candidate) => (
                                <option
                                  key={`${row.relation}-${candidate.entity ?? candidate.value}`}
                                  value={candidate.entity ?? candidate.value}
                                >
                                  {candidate.valueZh && candidate.valueEn
                                    ? `${candidate.valueZh} / ${candidate.valueEn}`
                                    : candidate.valueZh ||
                                      candidate.valueEn ||
                                      candidate.value}
                                </option>
                              ))}
                            </select>
                          ) : (
                            <div className="space-y-1">
                              <span>
                                {row.selectedValueZh ?? row.selectedValue}
                              </span>
                              {row.selectedValueEn ? (
                                <p className="text-xs text-muted-foreground">
                                  {row.selectedValueEn}
                                </p>
                              ) : null}
                            </div>
                          )}
                        </td>

                        <td className="p-3 align-middle">
                          {row.source === "Existing" ? (
                            <Badge variant="outline">Existing</Badge>
                          ) : (
                            <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100">
                              Predicted
                            </Badge>
                          )}
                        </td>

                        <td className="p-3 align-middle">
                          {row.source === "Existing" ? (
                            <span className="text-muted-foreground">-</span>
                          ) : (
                            <span>{formatScore(row.selectedScore)}</span>
                          )}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            <div className="flex gap-3 pt-2">
              <button
                onClick={handleNextProduct}
                className="flex-1 rounded-lg bg-primary px-4 py-2 text-primary-foreground hover:opacity-90 transition-opacity"
              >
                <span className="inline-flex items-center justify-center gap-2">
                  <ChevronRight className="w-4 h-4" />
                  Next Product
                </span>
              </button>

              <button
                onClick={handleRandomProduct}
                className="flex-1 rounded-lg border border-border px-4 py-2 hover:bg-muted transition-colors"
              >
                <span className="inline-flex items-center justify-center gap-2">
                  <Shuffle className="w-4 h-4" />
                  Random Product
                </span>
              </button>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
