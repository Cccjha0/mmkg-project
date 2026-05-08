export interface PerformanceOverview {
  best_accuracy_model: string;
  best_accuracy: number;
  num_models: number;
  last_updated: string;
  dataset_groups: {
    dataset: string;
    models: string[];
  }[];
}

export interface AccuracyCurveSeries {
  model: string;
  values: number[];
}

export interface AccuracyCurvesResponse {
  metric: string;
  datasets?: Record<
    string,
    {
      metric: string;
      dataset: string;
      epochs: number[];
      series: AccuracyCurveSeries[];
    }
  >;
  dataset?: string;
  epochs?: number[];
  series?: AccuracyCurveSeries[];
}

export interface ModelComparisonRow {
  model: string;
  dataset: string;
  accuracy: number;
  mrr: number;
  hits1: number;
  hits3: number;
  hits10: number;
  best_epoch: number;
}

export interface ModelComparisonResponse {
  metric: string;
  rows: ModelComparisonRow[];
}

export interface EntityResult {
  entity_id?: number;
  entity: string;
  entity_text?: string | null;
  entity_text_zh?: string | null;
  entity_text_en?: string | null;
  has_image: boolean;
  image_path: string | null;
  image_status?: string;
  has_text_embedding?: boolean;
  has_image_embedding?: boolean;
  available_spaces?: string[];
  gate_mean?: number;
  model_name?: string;
  dataset_name?: string;
}

export interface EntityResponse {
  task?: string;
  model?: string;
  device?: string;
  inputs?: {
    entity: string;
    entity_text?: string | null;
    entity_text_zh?: string | null;
    entity_text_en?: string | null;
  };
  results?: EntityResult;
  entity?: string;
  entity_text?: string | null;
  entity_text_zh?: string | null;
  entity_text_en?: string | null;
  has_image?: boolean;
  image_path?: string | null;
  image_status?: string;
  available_spaces?: string[];
  gate_mean?: number;
  model_name?: string;
  dataset_name?: string;
}

export interface AttributeOption {
  entity: string;
  entity_text_zh?: string | null;
  entity_text_en?: string | null;
  entity_text?: string | null;
  score: number | null;
  raw_score?: number | null;
  normalized_score?: number | null;
  display_score?: number | null;
  rank?: number | null;
}

export interface AttributeRowFromApi {
  relation?: string;
  relation_id?: string;
  relation_text_zh?: string | null;
  relation_text_en?: string | null;
  relation_text?: string | null;
  relation_name?: string | null;
  relation_name_en?: string | null;
  source: "existing" | "predicted" | string;
  selected_option_index: number;
  selected_value: AttributeOption;
  options: AttributeOption[];
  candidate_count?: number;
  warning?: string | null;
}

export interface AttributeCompletionResponse {
  task: string;
  model: string;
  device: string;
  inputs?: {
    entity: string;
    entity_text?: string | null;
    entity_text_zh?: string | null;
    entity_text_en?: string | null;
    topk: number;
  };
  results: {
    entity_info: {
      entity: string;
      entity_text?: string | null;
      entity_text_zh?: string | null;
      entity_text_en?: string | null;
      has_image: boolean;
      image_path: string | null;
    };
    attribute_rows: AttributeRowFromApi[];
  };
  latency_ms?: number;
}

export interface AttributeCandidate {
  entity?: string;
  value: string;
  valueZh?: string | null;
  valueEn?: string | null;
  score: number | null;
}

export interface AttributeDisplayRow {
  relation: string;
  relationText: string;
  relationTextZh?: string | null;
  relationTextEn?: string | null;
  source: "Existing" | "Predicted";
  selectedEntity?: string;
  selectedValue: string;
  selectedValueZh?: string | null;
  selectedValueEn?: string | null;
  selectedScore: number | null;
  candidates: AttributeCandidate[];
}
