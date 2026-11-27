export const REQUIRED_VIEW_TYPES = [
  "Perspective",
  "Front",
  "Rear",
  "Right",
  "Left",
  "Top",
  "Bottom",
] as const;

export const VIEW_TYPES = [
  "Perspective",
  "Front",
  "Rear",
  "Left",
  "Right",
  "Top",
  "Bottom",
  "Other",
] as const;
export type ViewType = (typeof VIEW_TYPES)[number];
export type RequiredViewType = (typeof REQUIRED_VIEW_TYPES)[number];

export type ImageStatus = "Uploaded" | "Queued" | "Processing" | "Review required" | "Completed" | "Failed";
export type ApprovalStatus = "pending" | "approved" | "rejected";
export type QualityStatus = "passed" | "review_required" | "failed";

export interface User {
  id: number;
  name: string;
  email: string;
}

export interface Project {
  id: number;
  name: string;
  description: string;
  status: string;
  created_at: string;
  updated_at: string;
  image_count: number;
  status_counts: Partial<Record<ImageStatus, number>>;
}

export interface Meta {
  max_upload_bytes: number;
  accepted_types: string[];
  storage: "blob" | "local";
}

export interface ResultUrls {
  drawing: string | null;
  svg: string | null;
  background_removed: string | null;
  mask: string | null;
  outline: string | null;
  original: string;
}

export interface AiResult {
  id: number;
  image_id: number;
  status: ImageStatus;
  quality_status?: QualityStatus;
  quality_warnings?: string[] | null;
  error_message: string | null;
  failed_stage: string | null;
  metrics: Record<string, unknown> | null;
  updated_at: string;
  urls: ResultUrls;
}

export interface Job {
  id: number;
  job_status: ImageStatus;
  error_message: string | null;
  failed_stage: string | null;
  retry_count: number;
}

export interface ImageItem {
  id: number;
  project_id: number;
  original_filename: string;
  mime_type: string;
  size_bytes: number;
  width: number;
  height: number;
  view_type: ViewType;
  processing_status: ImageStatus;
  approval_status: ApprovalStatus;
  is_active: boolean;
  created_at: string;
  original_url: string;
  result: AiResult | null;
  job: Job | null;
}

export interface DocumentMetadata {
  id: number;
  project_id: number;
  applicant_name: string;
  product_title: string;
  application_number: string | null;
  filing_date: string;
  novelty_statement: string;
  disclaimer: string;
  agent_name: string | null;
  agent_registration_number: string | null;
  agent_designation: string | null;
  has_signature: boolean;
  signature_url: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProjectDocument {
  id: number;
  project_id: number;
  version: number;
  filename: string;
  page_count: number;
  created_at: string;
  download_url: string;
}

export interface ProjectViews {
  project_id: number;
  views: Record<string, ImageItem | null>;
  other_images: ImageItem[];
  is_ready_to_process: boolean;
  is_ready_for_document: boolean;
  missing_views: string[];
  unapproved_views: string[];
  view_order: string[];
}

export interface ViewProcessStatus {
  image_id: number;
  view_type: string;
  status: ImageStatus;
  failed_stage?: string | null;
  error_message?: string | null;
  drawing_url?: string | null;
}

export interface ProjectProcessStatus {
  project_id: number;
  status: "idle" | "processing" | "completed" | "partial_failure" | "failed";
  total: number;
  completed: number;
  failed: number;
  in_progress: number;
  queued: number;
  current_stage?: string | null;
  current_view?: string | null;
  current_step?: number | null;
  error?: string | null;
  views: ViewProcessStatus[];
}

export interface Dashboard {
  project_count: number;
  image_count: number;
  status_counts: Partial<Record<ImageStatus, number>>;
  recent_projects: Project[];
  recent_results: ImageItem[];
}
