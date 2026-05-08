import { useCallback, useEffect, useMemo, useState } from "react";
import { getAttributeCompletion, getEntity } from "../lib/api";
import type {
  AttributeCandidate,
  AttributeCompletionResponse,
  AttributeDisplayRow,
  AttributeOption,
  AttributeRowFromApi,
  EntityResult,
  EntityResponse,
} from "../types/api";

interface UseAttributeCompletionReturn {
  entity: EntityResult | null;
  rows: AttributeDisplayRow[];
  loading: boolean;
  error: string | null;
  reload: () => Promise<void>;
  updatePredictedSelection: (relation: string, nextEntity: string) => void;
}

function getOptionText(option?: AttributeOption | null): string {
  if (!option) return "-";

  return (
    option.entity_text ||
    option.entity_text_zh ||
    option.entity_text_en ||
    option.entity ||
    "-"
  );
}

function normalizeSource(source: string): "Existing" | "Predicted" {
  return source?.toLowerCase() === "predicted" ? "Predicted" : "Existing";
}

function normalizeAttributeRows(
  apiRows?: AttributeRowFromApi[]
): AttributeDisplayRow[] {
  if (!apiRows || !Array.isArray(apiRows)) {
    return [];
  }

  return apiRows.map((row) => {
    const source = normalizeSource(row.source);
    const selectedValue = getOptionText(row.selected_value);
    const selectedEntity = row.selected_value?.entity;
    const relationKey = row.relation_id || row.relation || "";
    const relationText =
      row.relation_name ||
      row.relation_text ||
      row.relation_text_zh ||
      row.relation_name_en ||
      row.relation_text_en ||
      relationKey;
    const relationTextZh =
      row.relation_name || row.relation_text_zh || row.relation_text || null;
    const relationTextEn =
      row.relation_name_en || row.relation_text_en || null;

    const rawOptions =
      Array.isArray(row.options) && row.options.length > 0
        ? row.options
        : row.selected_value
        ? [row.selected_value]
        : [];

    const candidates: AttributeCandidate[] = rawOptions.map((option) => ({
      entity: option.entity,
      value: getOptionText(option),
      valueZh: option.entity_text_zh ?? option.entity_text ?? null,
      valueEn: option.entity_text_en ?? null,
      score: option.score ?? null,
    }));

    const matchedCandidate =
      candidates.find((candidate) => candidate.entity === selectedEntity) ??
      candidates.find((candidate) => candidate.value === selectedValue) ??
      null;

    const selectedScore =
      source === "Predicted"
        ? matchedCandidate?.score ?? row.selected_value?.score ?? null
        : null;

    return {
      relation: relationKey,
      relationText,
      relationTextZh,
      relationTextEn,
      source,
      selectedEntity,
      selectedValue,
      selectedValueZh:
        row.selected_value?.entity_text_zh ??
        row.selected_value?.entity_text ??
        null,
      selectedValueEn: row.selected_value?.entity_text_en ?? null,
      selectedScore,
      candidates,
    };
  });
}

function normalizeEntity(
  entityData: EntityResponse,
  attributeData: AttributeCompletionResponse
): EntityResult {
  const entityFromDetail = entityData.results ?? {
    entity: entityData.entity ?? "",
    entity_text: entityData.entity_text ?? null,
    entity_text_zh: entityData.entity_text_zh ?? null,
    entity_text_en: entityData.entity_text_en ?? null,
    has_image: entityData.has_image ?? false,
    image_path: entityData.image_path ?? null,
    image_status: entityData.image_status,
    available_spaces: entityData.available_spaces,
    gate_mean: entityData.gate_mean,
    model_name: entityData.model_name,
    dataset_name: entityData.dataset_name,
  };

  const entityFromAttribute = attributeData.results.entity_info;

  return {
    ...entityFromDetail,
    ...entityFromAttribute,
    entity: entityFromAttribute.entity || entityFromDetail.entity,
    entity_text:
      entityFromAttribute.entity_text ?? entityFromDetail.entity_text ?? null,
    entity_text_zh:
      entityFromAttribute.entity_text_zh ??
      entityFromDetail.entity_text_zh ??
      null,
    entity_text_en:
      entityFromAttribute.entity_text_en ??
      entityFromDetail.entity_text_en ??
      null,
    has_image: entityFromAttribute.has_image ?? entityFromDetail.has_image,
    image_path: entityFromAttribute.image_path ?? entityFromDetail.image_path,
  };
}

export function useAttributeCompletion(
  entityId: string
): UseAttributeCompletionReturn {
  const [entity, setEntity] = useState<EntityResult | null>(null);
  const [rowsState, setRowsState] = useState<AttributeDisplayRow[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    if (!entityId) {
      setEntity(null);
      setRowsState([]);
      setLoading(false);
      setError("Entity ID is empty.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const [entityData, attributeData] = await Promise.all([
        getEntity(entityId),
        getAttributeCompletion(entityId),
      ]);

      setEntity(normalizeEntity(entityData, attributeData));
      setRowsState(normalizeAttributeRows(attributeData.results.attribute_rows));
    } catch (err) {
      console.error("Failed to load attribute completion data:", err);
      setError("Failed to load attribute completion data.");
      setEntity(null);
      setRowsState([]);
    } finally {
      setLoading(false);
    }
  }, [entityId]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  const updatePredictedSelection = useCallback(
    (relation: string, nextEntity: string) => {
      setRowsState((prevRows) =>
        prevRows.map((row) => {
          if (row.relation !== relation || row.source !== "Predicted") {
            return row;
          }

          const matched = row.candidates.find(
            (candidate) => candidate.entity === nextEntity
          );

          return {
            ...row,
            selectedEntity: nextEntity,
            selectedValue: matched?.value ?? nextEntity,
            selectedValueZh: matched?.valueZh ?? null,
            selectedValueEn: matched?.valueEn ?? null,
            selectedScore: matched?.score ?? null,
          };
        })
      );
    },
    []
  );

  const rows = useMemo(() => rowsState, [rowsState]);

  return {
    entity,
    rows,
    loading,
    error,
    reload: loadData,
    updatePredictedSelection,
  };
}
