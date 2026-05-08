import type {
  AccuracyCurvesResponse,
  AttributeCompletionResponse,
  EntityResponse,
  ModelComparisonResponse,
  PerformanceOverview,
} from "../types/api";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const API_PREFIX = `${API_BASE_URL}/api`;

export function resolveAssetUrl(path?: string | null): string | null {
  if (!path) return null;

  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }

  if (path.startsWith("/")) {
    return `${API_BASE_URL}${path}`;
  }

  return `${API_BASE_URL}/${path}`;
}

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_PREFIX}${path}`);

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(
      `Request failed: ${response.status} ${response.statusText} ${errorText}`
    );
  }

  return response.json() as Promise<T>;
}

export async function getPerformanceOverview(): Promise<PerformanceOverview> {
  return request<PerformanceOverview>("/performance/overview");
}

export async function getAccuracyCurves(
  dataset = "OpenBG-IMG"
): Promise<AccuracyCurvesResponse> {
  return request<AccuracyCurvesResponse>(
    `/performance/accuracy-curves?dataset=${encodeURIComponent(dataset)}`
  );
}

export async function getModelComparison(
  dataset = "OpenBG-IMG"
): Promise<ModelComparisonResponse> {
  return request<ModelComparisonResponse>(
    `/performance/model-comparison?dataset=${encodeURIComponent(dataset)}`
  );
}

export async function getEntity(entityId: string): Promise<EntityResponse> {
  return request<EntityResponse>(`/entities/${encodeURIComponent(entityId)}`);
}

export async function getAttributeCompletion(
  entityId: string,
  topk = 5
): Promise<AttributeCompletionResponse> {
  return request<AttributeCompletionResponse>(
    `/entities/${encodeURIComponent(entityId)}/attribute-completion?topk=${topk}`
  );
}