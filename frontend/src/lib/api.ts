import type {
  Dashboard,
  DocumentMetadata,
  ImageItem,
  Meta,
  Project,
  ProjectDocument,
  ProjectViews,
  User,
  ViewType,
} from "./types";

/** Same-origin by default: Vercel routes /api/* to the FastAPI service; `next dev` proxies it locally. */
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "/api";
const TOKEN_KEY = "pv_token";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code: string,
    public nextAction?: string,
  ) {
    super(message);
  }
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null): void {
  if (token) window.localStorage.setItem(TOKEN_KEY, token);
  else window.localStorage.removeItem(TOKEN_KEY);
}

/** Turn a failed response into an ApiError using the backend's structured error body. */
export async function toApiError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    const e = body?.error;
    if (e?.message) {
      const detail = Array.isArray(e.details) && e.details.length ? ` (${e.details[0].field}: ${e.details[0].message})` : "";
      return new ApiError(e.message + detail, res.status, e.code ?? "error", e.next_action);
    }
  } catch {
    /* fall through */
  }
  return new ApiError(`Request failed (${res.status}).`, res.status, "error");
}

async function send(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { ...init, headers });
  } catch {
    throw new ApiError("Cannot reach the PatentVision server.", 0, "network", "Check that the backend is running.");
  }
  if (!res.ok) throw await toApiError(res);
  return res;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await send(path, init);
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

const body = (data: unknown): RequestInit => ({ body: JSON.stringify(data) });

export const api = {
  register: (name: string, email: string, password: string) =>
    json<{ access_token: string; user: User }>("/auth/register", { method: "POST", ...body({ name, email, password }) }),
  login: (email: string, password: string) =>
    json<{ access_token: string; user: User }>("/auth/login", { method: "POST", ...body({ email, password }) }),
  me: () => json<User>("/auth/me"),
  meta: () => json<Meta>("/meta"),

  dashboard: () => json<Dashboard>("/dashboard"),
  listProjects: () => json<Project[]>("/projects"),
  getProject: (id: number) => json<Project>(`/projects/${id}`),
  createProject: (name: string, description: string) =>
    json<Project>("/projects", { method: "POST", ...body({ name, description }) }),
  updateProject: (id: number, patch: Partial<Pick<Project, "name" | "description" | "status">>) =>
    json<Project>(`/projects/${id}`, { method: "PUT", ...body(patch) }),
  deleteProject: (id: number) => json<void>(`/projects/${id}`, { method: "DELETE" }),

  listImages: (projectId: number) => json<ImageItem[]>(`/projects/${projectId}/images`),
  getImage: (id: number) => json<ImageItem>(`/images/${id}`),
  uploadImage: (projectId: number, file: File, view: ViewType) => {
    const form = new FormData();
    form.append("file", file);
    form.append("view_type", view);
    return json<ImageItem>(`/projects/${projectId}/images`, { method: "POST", body: form });
  },
  setView: (id: number, view: ViewType) =>
    json<ImageItem>(`/images/${id}`, { method: "PATCH", ...body({ view_type: view }) }),
  deleteImage: (id: number) => json<void>(`/images/${id}`, { method: "DELETE" }),

  // 7-view slot APIs
  getProjectViews: (projectId: number) => json<ProjectViews>(`/projects/${projectId}/views`),
  uploadView: (projectId: number, view: ViewType, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return json<ImageItem>(`/projects/${projectId}/views/${view}`, { method: "POST", body: form });
  },
  deleteView: (projectId: number, view: ViewType) =>
    json<void>(`/projects/${projectId}/views/${view}`, { method: "DELETE" }),

  // Review & Approval APIs
  reviewImage: (imageId: number, action: "approve" | "reject", notes?: string) =>
    json<ImageItem>(`/images/${imageId}/review`, { method: "POST", ...body({ action, notes }) }),

  // Document metadata & generation APIs
  getDocumentMetadata: (projectId: number) =>
    json<DocumentMetadata>(`/projects/${projectId}/document-metadata`),
  updateDocumentMetadata: (projectId: number, data: Partial<DocumentMetadata>) =>
    json<DocumentMetadata>(`/projects/${projectId}/document-metadata`, { method: "PUT", ...body(data) }),
  uploadSignature: (projectId: number, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return json<DocumentMetadata>(`/projects/${projectId}/document-metadata/signature`, {
      method: "POST",
      body: form,
    });
  },
  deleteSignature: (projectId: number) =>
    json<void>(`/projects/${projectId}/document-metadata/signature`, { method: "DELETE" }),
  generateDocument: (projectId: number) =>
    json<{ project_id: number; document: ProjectDocument }>(`/projects/${projectId}/documents`, {
      method: "POST",
    }),
  listDocuments: (projectId: number) =>
    json<ProjectDocument[]>(`/projects/${projectId}/documents`),

  processImage: (id: number) =>
    json<{ image_id: number; status: ImageItem["processing_status"] }>(`/ai/process/${id}`, { method: "POST" }),
  processProject: (id: number) => json<{ queued: number[]; skipped: number[] }>(`/ai/process/project/${id}`, { method: "POST" }),
};

/** Fetch a protected file (images need the bearer token, so <img src> cannot be used directly). */
export async function fetchBlob(urlPath: string): Promise<Blob> {
  const res = await send(urlPath);
  return res.blob();
}

export async function downloadFile(urlPath: string, filename: string): Promise<void> {
  const blob = await fetchBlob(urlPath);
  const href = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = href;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(href);
}
