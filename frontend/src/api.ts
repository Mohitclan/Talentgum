// Typed client for the RAS API (/v1). Shapes follow ras/service/api.py and services.assessment_view.

export type Status = "queued" | "processing" | "completed" | "review_required" | "reviewed" | "rejected" | "failed";

export interface CriterionScore {
  weight: number;
  metric: string;
  value: number | null;
  score: number | null;
  status: "assessed" | "not_assessed";
  reason?: string;
}

export interface Assessment {
  assessment_id: string;
  status: Status;
  song_id: string;
  reference_version: number;
  rubric: { id: string; version: number };
  student_id: string | null;
  external_ref: string | null;
  attempts: number;
  error_code: string | null;
  error_detail: string | null;
  needs_review: boolean;
  total_score: number | null;
  confidence: number | null;
  created_at: string;
  finished_at: string | null;
  recording_file_id: string;
  summary?: Record<string, any>;
  scoring?: { criteria: Record<string, CriterionScore>; total?: number | null; band?: string | null; [k: string]: any };
  confidence_detail?: { value: number; needs_review: boolean; reasons: string[] };
  feedback?: string[];
  warnings?: string[];
  quality?: Record<string, any>;
}

export type NoteStatus =
  | "correct" | "wrong_pitch" | "wrong_octave" | "missing" | "uncertain"
  | "extra" | "unconfirmed_extra" | "replayed";

export interface Note {
  status: NoteStatus;
  ref_index: number | null;
  perf_index: number | null;
  measure: number | null;
  beat_in_measure: number | null;
  ref_pitch: number | null;
  perf_pitch: number | null;
  perf_onset_s: number | null;
  onset_dev_ms: number | null;
  duration_ratio: number | null;
  expected_onset_s: number | null;
}

export interface Song {
  song_id: string;
  title: string;
  references: { version: number; notes: number; measures: number; bpm: number | null; warnings: string[] }[];
}

export interface Review {
  review_id: string;
  reviewer: string;
  corrections: any[];
  teacher_scores: Record<string, number>;
  overall_score: number | null;
  comment: string | null;
  created_at: string;
}

export interface Criterion { id: string; group: string; label: string }
export interface RubricInfo { rubric_id: string; version: number; criteria: string[] }

export class ApiError extends Error {
  constructor(public status: number, public code: string, detail: string) {
    super(detail);
  }
}

const KEY_STORE = "tg.apiKey";

export function getApiKey(): string {
  try { return localStorage.getItem(KEY_STORE) ?? ""; } catch { return ""; }
}

export function setApiKey(key: string) {
  try { key ? localStorage.setItem(KEY_STORE, key) : localStorage.removeItem(KEY_STORE); } catch { /* private mode */ }
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const key = getApiKey();
  return key ? { "X-API-Key": key, ...extra } : extra;
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`/v1${path}`, { ...init, headers: headers(init.headers as Record<string, string>) });
  if (!res.ok) {
    let code = "http_error", detail = res.statusText;
    try { const p = await res.json(); code = p.code ?? code; detail = p.detail ?? detail; } catch { /* not JSON */ }
    throw new ApiError(res.status, code, detail);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST", body: JSON.stringify(body), headers: { "Content-Type": "application/json" },
});

export const api = {
  health: () => call<{ status: string; transcriber: string; auth: string }>("/health"),
  criteria: () => call<Criterion[]>("/criteria"),
  songs: () => call<Song[]>("/songs"),
  rubrics: () => call<RubricInfo[]>("/rubrics"),
  assessments: (q: { status?: string; student_id?: string; limit?: number } = {}) => {
    const p = new URLSearchParams();
    Object.entries(q).forEach(([k, v]) => v !== undefined && v !== "" && p.set(k, String(v)));
    return call<Assessment[]>(`/assessments?${p}`);
  },
  assessment: (id: string) => call<Assessment>(`/assessments/${id}`),
  notes: (id: string) => call<{ notes: Note[] }>(`/assessments/${id}/notes`).then((r) => r.notes),
  reviews: (id: string) => call<Review[]>(`/assessments/${id}/reviews`),
  review: (id: string, body: { reviewer: string; overall_score?: number | null; comment?: string; teacher_scores?: Record<string, number> }) =>
    call<{ review_id: string }>(`/assessments/${id}/review`, json(body)),
  submit: (body: { song_id: string; recording_file_id: string; rubric_id: string; student_id: string }, idempotencyKey: string) =>
    call<{ assessment_id: string; status: Status }>("/assessments", {
      ...json(body), headers: { "Content-Type": "application/json", "Idempotency-Key": idempotencyKey },
    }),
  recording: async (fileId: string): Promise<string> => {
    const res = await fetch(`/v1/files/${fileId}/content`, { headers: headers() });
    if (!res.ok) throw new ApiError(res.status, "http_error", "could not load the recording");
    return URL.createObjectURL(await res.blob());
  },
};

/** Upload with progress (fetch cannot report upload progress). */
export function uploadAudio(file: File, onProgress: (pct: number) => void): Promise<{ file_id: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/v1/files?kind=audio");
    Object.entries(headers()).forEach(([k, v]) => xhr.setRequestHeader(k, v));
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress((e.loaded / e.total) * 100);
    xhr.onload = () => {
      let body: any = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status === 201) resolve(body);
      else reject(new ApiError(xhr.status, body.code ?? "upload_failed", body.detail ?? "Upload failed"));
    };
    xhr.onerror = () => reject(new ApiError(0, "network", "Network error during upload"));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}
