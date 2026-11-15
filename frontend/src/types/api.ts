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
export interface ThreatFeedItem {
  title: string;
  url: string;
  domain: string | null;
  seendate: string | null;
  socialimage: string | null;
  language: string | null;
  sourcecountry: string | null;
}

export interface ThreatFeedResponse {
  items: ThreatFeedItem[];
  cached_at: string;
  source: string;
}

export interface ResearchPaperItem {
  arxiv_id: string;
  title: string;
  authors: string[];
  summary: string;
  url: string;
  published_at: string;
}

export interface ResearchFeedResponse {
  items: ResearchPaperItem[];
  cached_at: string;
  source: string;
}

export interface PulseFeedItem {
  object_id: string;
  title: string;
  url: string | null;
  author: string;
  points: number;
  num_comments: number;
  created_at: string;
}

export interface PulseFeedResponse {
  items: PulseFeedItem[];
  cached_at: string;
  source: string;
}

export interface SampleItem {
  id: string;
  title: string;
  description: string;
  media_type: 'video' | 'audio';
  duration_s: number;
  file_size_bytes: number;
  source: string;
  page_url: string;
  download_url: string;
  license: string;
  attribution: string;
  ground_truth: 'authentic' | 'synthetic' | 'inconclusive';
  thumbnail_url: string | null;
  context_summary: string | null;
}

export interface SampleListResponse {
  items: SampleItem[];
  cached_at: string;
  total: number;
}

export interface SampleAnalyzeResponse {
  id: string;
  status: string;
  sample_id: string;
}
