// TypeScript mirrors of backend Pydantic schemas.
// Keep in sync with backend/app/schemas/*.py

export interface UserResponse {
  id: number;
  email: string;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: 'bearer';
  user: UserResponse;
}

export interface MessageResponse {
  message: string;
}

export interface HealthResponse {
  status: 'ok';
}

export interface ReadyResponse {
  status: 'ready' | 'degraded';
  db_connected: boolean;
  model_loaded: boolean;
  model_mode: 'trained' | 'untrained';
}

export interface ModelCardMetrics {
  accuracy_controlled: number;
  accuracy_real_world: number;
  auc_roc: number;
  source: string;
}

export interface ModelCardResponse {
  model_name: string;
  model_version: string;
  model_mode: 'trained' | 'untrained';
  checkpoint_hash: string | null;
  reported_metrics: ModelCardMetrics;
  live_metrics: Record<string, number> | null;
}

export interface ApiError {
  code: string;
  message: string;
}

// Analysis schemas (Phase 6 — referenced here for type completeness)
export type AnalysisStatus = 'queued' | 'running' | 'done' | 'error';
export type AnalysisStage =
  | 'probe'
  | 'frames'
  | 'faces'
  | 'audio'
  | 'align'
  | 'face_enc'
  | 'audio_enc'
  | 'fusion'
  | 'explain'
  | 'done';
export type Verdict = 'AUTHENTIC' | 'FAKE' | 'INCONCLUSIVE';

export interface AnalysisCreated {
  id: string;
  status: AnalysisStatus;
}

export interface TimelinePoint {
  t: number;
  p_face: number;
  p_audio: number;
  p_fused: number;
}

export interface ArtifactItem {
  id: number;
  kind: string;
  name: string;
}

export interface AnalysisSummary {
  id: string;
  original_name: string;
  status: AnalysisStatus;
  stage: string;
  verdict: Verdict | null;
  p_fake: number | null;
  duration_s: number;
  created_at: string;
  completed_at: string | null;
}

export interface AnalysisDetail extends AnalysisSummary {
  w_audio: number | null;
  w_video: number | null;
  sync_score: number | null;
  face_coverage: number | null;
  audio_present: boolean;
  model_version: string;
  model_mode: 'trained' | 'untrained';
  timeline: TimelinePoint[] | null;
  reasons: string[] | null;
  error: string | null;
  artifacts: ArtifactItem[];
}

export interface AnalysisListResponse {
  items: AnalysisSummary[];
  total: number;
  page: number;
  size: number;
}

export interface StageEventPayload {
  stage: string;
  progress: number;
  message: string;
  elapsed_s: number;
}

export interface SSEEvent {
  stage: AnalysisStage;
  progress: number;
  message: string;
}

// Feed schemas (Phase 8)
export interface FeedItem {
  title: string;
  source: string;
  url: string;
  published_at: string;
}

export interface SampleItem {
  id: string;
  title: string;
  source: string;
  url: string;
  licence: string;
  attribution: string;
  duration_s: number | null;
  thumbnail_url: string | null;
}
