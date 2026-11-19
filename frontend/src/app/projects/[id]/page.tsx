"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { DropZone } from "@/components/DropZone";
import { ErrorBanner } from "@/components/ErrorBanner";
import { SecureImage } from "@/components/SecureImage";
import { StatusBadge } from "@/components/StatusBadge";
import { useToast } from "@/components/Toast";
import { usePolling } from "@/hooks/usePolling";
import { api, downloadFile, fetchBlob } from "@/lib/api";
import {
  checkUploadFile,
  formatBytes,
  formatDate,
  isActive,
  processActionLabel,
  stageLabel,
} from "@/lib/format";
import { prepareForUpload } from "@/lib/imageprep";
import {
  REQUIRED_VIEW_TYPES,
  type DocumentMetadata,
  type ImageItem,
  type Project,
  type ProjectDocument,
  type RequiredViewType,
} from "@/lib/types";
import {
  detectViewTypeFromFilename,
  REQUIRED_SLOTS,
  validateProjectViewsReadiness,
  type SlotConfig,
} from "@/lib/views";

type QueueState = "waiting" | "optimizing" | "uploading" | "done" | "error";
interface QueueItem {
  key: string;
  name: string;
  state: QueueState;
  note?: string;
}

const QUEUE_LABEL: Record<QueueState, string> = {
  waiting: "Waiting",
  optimizing: "Optimizing",
  uploading: "Uploading",
  done: "Uploaded",
  error: "Failed",
};

function ProjectView({ id }: { id: number }) {
  const router = useRouter();
  const toast = useToast();
  const [project, setProject] = useState<Project | null>(null);
  const [views, setViews] = useState<Record<RequiredViewType, ImageItem | null>>({
    Perspective: null,
    Front: null,
    Rear: null,
    Right: null,
    Left: null,
    Top: null,
    Bottom: null,
  });
  const [metadata, setMetadata] = useState<DocumentMetadata | null>(null);
  const [documents, setDocuments] = useState<ProjectDocument[]>([]);
  const [metaForm, setMetaForm] = useState<Partial<DocumentMetadata>>({});
  const [error, setError] = useState<unknown>(null);
  const [autoProcess, setAutoProcess] = useState(true);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [busy, setBusy] = useState<Set<number>>(new Set());
  const [batch, setBatch] = useState<{ done: number; total: number } | null>(null);
  const [maxBytes, setMaxBytes] = useState(4 * 1024 * 1024);
  const [editingProject, setEditingProject] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [projectDesc, setProjectDesc] = useState("");
  const [generatingPdf, setGeneratingPdf] = useState(false);
  const [savingMeta, setSavingMeta] = useState(false);
  const [sigUploading, setSigUploading] = useState(false);

  const signatureInputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(() => {
    Promise.all([
      api.getProject(id),
      api.getProjectViews(id),
      api.getDocumentMetadata(id),
      api.listDocuments(id),
    ])
      .then(([p, v, m, docs]) => {
        setProject(p);
        setViews(v.views);
        setMetadata(m);
        setDocuments(docs);
        setMetaForm((prev) => ({
          applicant_name: prev.applicant_name ?? m.applicant_name ?? "",
          product_title: prev.product_title ?? m.product_title ?? p.name,
          application_number: prev.application_number ?? m.application_number ?? "",
          filing_date: prev.filing_date ?? m.filing_date ?? "",
          agent_name: prev.agent_name ?? m.agent_name ?? "",
          agent_registration_number: prev.agent_registration_number ?? m.agent_registration_number ?? "",
          agent_designation: prev.agent_designation ?? m.agent_designation ?? "Agent for the Applicant",
          novelty_statement: prev.novelty_statement ?? m.novelty_statement ?? "",
          disclaimer: prev.disclaimer ?? m.disclaimer ?? "",
        }));
      })
      .catch(setError);
  }, [id]);

  useEffect(load, [load]);

  useEffect(() => {
    api.meta().then((m) => setMaxBytes(m.max_upload_bytes)).catch(() => {});
  }, []);

  const activeImages = Object.values(views).filter(
    (img): img is ImageItem => img !== null && isActive(img.processing_status)
  );
  usePolling(load, activeImages.length > 0 && busy.size === 0 && !batch, 2000);

  const setItem = (key: string, patch: Partial<QueueItem>) =>
    setQueue((q) => q.map((it) => (it.key === key ? { ...it, ...patch } : it)));

  async function processImageById(imageId: number): Promise<boolean> {
    setBusy((b) => new Set(b).add(imageId));
    try {
      const r = await api.processImage(imageId);
      return r.status === "Completed" || r.status === "Review required";
    } catch (err) {
      setError(err);
      return false;
    } finally {
      setBusy((b) => {
        const next = new Set(b);
        next.delete(imageId);
        return next;
      });
      load();
    }
  }

  async function runBatch(imagesToProcess: ImageItem[]) {
    if (imagesToProcess.length === 0) return;
    setBatch({ done: 0, total: imagesToProcess.length });
    const pool = [...imagesToProcess];
    let ok = 0;
    let completed = 0;

    const worker = async () => {
      while (pool.length > 0) {
        const item = pool.shift();
        if (!item) break;
        const res = await processImageById(item.id);
        if (res) ok++;
        completed++;
        setBatch({ done: completed, total: imagesToProcess.length });
      }
    };

    await Promise.all([worker(), worker()]);
    setBatch(null);
    load();
    if (ok === imagesToProcess.length) {
      toast("success", `All ${imagesToProcess.length} view drawings generated.`);
    } else {
      toast("info", `Processed ${ok} of ${imagesToProcess.length} view drawings.`);
    }
  }

  async function uploadBatchSession(files: File[]) {
    setError(null);
    if (!files.length) return;

    const assignments: { file: File; viewType: RequiredViewType; key: string }[] = [];
    const usedSlots = new Set<RequiredViewType>();

    // Pass 1: detect by filename
    for (const file of files) {
      const detected = detectViewTypeFromFilename(file.name);
      if (detected && !usedSlots.has(detected)) {
        usedSlots.add(detected);
        assignments.push({ file, viewType: detected, key: `${Date.now()}-${file.name}` });
      }
    }

    // Pass 2: assign remaining files to unfilled slots
    const unassigned = files.filter((f) => !assignments.some((a) => a.file === f));
    for (const file of unassigned) {
      const slot = REQUIRED_VIEW_TYPES.find((s) => !usedSlots.has(s) && views[s] === null);
      const fallback = REQUIRED_VIEW_TYPES.find((s) => !usedSlots.has(s)) ?? "Perspective";
      const chosen = slot ?? fallback;
      usedSlots.add(chosen);
      assignments.push({ file, viewType: chosen, key: `${Date.now()}-${file.name}` });
    }

    const items: QueueItem[] = assignments.map((a) => ({
      key: a.key,
      name: `${a.file.name} → ${a.viewType}`,
      state: "waiting",
    }));
    setQueue((q) => [...items, ...q.filter((it) => it.state !== "done")].slice(0, 14));

    const uploadedImgs: ImageItem[] = [];
    for (const { file, viewType, key } of assignments) {
      const problem = checkUploadFile(file);
      if (problem) {
        setItem(key, { state: "error", note: problem });
        continue;
      }
      try {
        setItem(key, { state: "optimizing" });
        const prepared = await prepareForUpload(file, maxBytes);
        setItem(key, {
          state: "uploading",
          note: prepared.optimized ? `Optimized ${formatBytes(prepared.originalSize)} → ${formatBytes(prepared.file.size)}` : undefined,
        });
        const img = await api.uploadView(id, viewType, prepared.file);
        uploadedImgs.push(img);
        setItem(key, { state: "done", note: `Slot: ${viewType}` });
      } catch (err) {
        setItem(key, { state: "error", note: err instanceof Error ? err.message : String(err) });
      }
    }

    load();
    if (uploadedImgs.length > 0) {
      toast("success", `Uploaded ${uploadedImgs.length} photo${uploadedImgs.length > 1 ? "s" : ""} to assigned slots.`);
      if (autoProcess) {
        await runBatch(uploadedImgs);
      }
    }
  }

  async function uploadToSlot(file: File, viewType: RequiredViewType) {
    setError(null);
    const problem = checkUploadFile(file);
    if (problem) {
      toast("error", problem);
      return;
    }
    const key = `${Date.now()}-${file.name}`;
    setQueue((q) => [{ key, name: `${file.name} → ${viewType}`, state: "optimizing" }, ...q]);
    try {
      const prepared = await prepareForUpload(file, maxBytes);
      setItem(key, {
        state: "uploading",
        note: prepared.optimized ? `Optimized ${formatBytes(prepared.originalSize)} → ${formatBytes(prepared.file.size)}` : undefined,
      });
      const img = await api.uploadView(id, viewType, prepared.file);
      setItem(key, { state: "done", note: `Slot: ${viewType}` });
      toast("success", `${viewType} view uploaded.`);
      load();
      if (autoProcess) {
        await processImageById(img.id);
      }
    } catch (err) {
      setItem(key, { state: "error", note: err instanceof Error ? err.message : String(err) });
      setError(err);
    }
  }

  async function removeSlot(viewType: RequiredViewType) {
    const current = views[viewType];
    if (!current) return;
    if (!confirm(`Delete ${viewType} view photo (${current.original_filename}) and drawings?`)) return;
    try {
      await api.deleteView(id, viewType);
      toast("info", `${viewType} view removed.`);
      load();
    } catch (err) {
      setError(err);
    }
  }

  async function reviewSlot(imageId: number, action: "approve" | "reject") {
    try {
      await api.reviewImage(imageId, action);
      toast(action === "approve" ? "success" : "info", action === "approve" ? "Drawing approved." : "Drawing rejected.");
      load();
    } catch (err) {
      setError(err);
    }
  }

  async function saveProjectDetails(e: React.FormEvent) {
    e.preventDefault();
    try {
      setProject(await api.updateProject(id, { name: projectName, description: projectDesc }));
      setEditingProject(false);
      toast("success", "Project updated.");
    } catch (err) {
      setError(err);
    }
  }

  async function removeProject() {
    if (!confirm("Delete this project, all 7 views and generated documents? This cannot be undone.")) return;
    try {
      await api.deleteProject(id);
      router.replace("/");
    } catch (err) {
      setError(err);
    }
  }

  async function saveMetadata(e: React.FormEvent) {
    e.preventDefault();
    setSavingMeta(true);
    try {
      const updated = await api.updateDocumentMetadata(id, metaForm);
      setMetadata(updated);
      toast("success", "Application metadata saved.");
    } catch (err) {
      setError(err);
    } finally {
      setSavingMeta(false);
    }
  }

  async function handleSignatureFile(file: File) {
    setSigUploading(true);
    try {
      const updated = await api.uploadSignature(id, file);
      setMetadata(updated);
      toast("success", "Signature uploaded.");
    } catch (err) {
      setError(err);
    } finally {
      setSigUploading(false);
    }
  }

  async function handleSignatureRemove() {
    if (!confirm("Remove the signature file?")) return;
    setSigUploading(true);
    try {
      await api.deleteSignature(id);
      setMetadata((prev) => (prev ? { ...prev, has_signature: false, signature_url: null } : null));
      toast("info", "Signature removed.");
    } catch (err) {
      setError(err);
    } finally {
      setSigUploading(false);
    }
  }

  async function handleGeneratePdf() {
    setError(null);
    setGeneratingPdf(true);
    try {
      await api.generateDocument(id);
      toast("success", "Official 7-Page Patent Representation PDF generated!");
      load();
    } catch (err) {
      setError(err);
    } finally {
      setGeneratingPdf(false);
    }
  }

  async function handleDownload(doc: ProjectDocument) {
    try {
      await downloadFile(`/documents/${doc.id}/download`, doc.filename);
    } catch (err) {
      setError(err);
    }
  }

  async function handlePreview(doc: ProjectDocument) {
    try {
      const blob = await fetchBlob(`/documents/${doc.id}/preview`);
      const url = URL.createObjectURL(blob);
      window.open(url, "_blank");
    } catch (err) {
      setError(err);
    }
  }

  if (!project) {
    return error ? <ErrorBanner error={error} /> : <div className="h-40 animate-pulse rounded-2xl bg-slate-200/60" />;
  }

  const uploadedCount = REQUIRED_VIEW_TYPES.filter((v) => views[v] !== null).length;
  const drawingCount = REQUIRED_VIEW_TYPES.filter((v) => views[v]?.result?.urls.drawing).length;
  const approvedCount = REQUIRED_VIEW_TYPES.filter((v) => views[v]?.approval_status === "approved").length;
  const allApproved = approvedCount === 7;
  const pendingProcessing = REQUIRED_VIEW_TYPES.map((v) => views[v]).filter(
    (img): img is ImageItem => img !== null && !isActive(img.processing_status) && !busy.has(img.id) && img.processing_status !== "Completed"
  );

  return (
    <div className="space-y-8">
      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {/* Project Header */}
      <div>
        <Link href="/" className="text-sm text-slate-500 hover:text-slate-900">← All projects</Link>
        {editingProject ? (
          <form onSubmit={saveProjectDetails} className="mt-3 flex flex-col gap-2 rounded-2xl border border-slate-200 bg-white p-4 sm:flex-row">
            <input
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              aria-label="Project name"
              required
              className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm"
            />
            <input
              value={projectDesc}
              onChange={(e) => setProjectDesc(e.target.value)}
              aria-label="Description"
              placeholder="Notes or patent reference"
              className="flex-[1.4] rounded-lg border border-slate-300 px-3 py-2 text-sm"
            />
            <div className="flex gap-2">
              <button className="rounded-lg bg-slate-900 px-4 py-2 text-sm text-white">Save</button>
              <button
                type="button"
                onClick={() => setEditingProject(false)}
                className="rounded-lg px-3 py-2 text-sm text-slate-600 hover:bg-slate-100"
              >
                Cancel
              </button>
            </div>
          </form>
        ) : (
          <div className="mt-2 flex flex-wrap items-start justify-between gap-3">
            <div>
              <h1 className="text-3xl font-semibold tracking-tight">{project.name}</h1>
              <p className="mt-1 text-sm text-slate-500">
                {project.description || "No notes"} · {uploadedCount}/7 views uploaded · {drawingCount}/7 drawings ready · {approvedCount}/7 approved
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => {
                  setProjectName(project.name);
                  setProjectDesc(project.description);
                  setEditingProject(true);
                }}
                className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm hover:bg-slate-50"
              >
                Edit
              </button>
              <button onClick={removeProject} className="rounded-lg border border-red-200 bg-white px-3 py-1.5 text-sm text-red-700 hover:bg-red-50">
                Delete
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Multi-Photo Session Upload */}
      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-slate-900">Upload 7 Patent Views in One Session</h2>
            <p className="text-sm text-slate-500">
              Drop all 7 product photos at once. Photos matching &quot;perspective&quot;, &quot;front&quot;, &quot;rear&quot;, &quot;right&quot;, &quot;left&quot;, &quot;top&quot;, or &quot;bottom&quot; are automatically assigned to their respective slots.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-4 text-sm">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={autoProcess}
                onChange={(e) => setAutoProcess(e.target.checked)}
                className="h-4 w-4 rounded text-blue-600"
              />
              Automatically generate drawings after upload
            </label>
          </div>
        </div>

        <DropZone
          onFiles={uploadBatchSession}
          disabled={queue.some((q) => q.state === "optimizing" || q.state === "uploading")}
          hint={`JPG, PNG or WEBP. Photos over ${formatBytes(maxBytes)} are automatically optimized before upload.`}
        />

        {queue.length > 0 && (
          <ul className="mt-4 divide-y divide-slate-100 rounded-lg border border-slate-100 text-sm">
            {queue.map((q) => (
              <li key={q.key} className="flex items-center justify-between gap-3 px-3 py-2">
                <span className="truncate">{q.name}</span>
                <span className="flex shrink-0 items-center gap-2 text-xs">
                  {q.note && <span className={q.state === "error" ? "text-red-700" : "text-slate-500"}>{q.note}</span>}
                  <span
                    className={
                      q.state === "done"
                        ? "text-emerald-700 font-medium"
                        : q.state === "error"
                          ? "font-medium text-red-700"
                          : "text-blue-700"
                    }
                  >
                    {QUEUE_LABEL[q.state]}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* Global Processing Action Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-slate-700">Views Status:</span>
          <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold ${
            uploadedCount === 7 ? "bg-emerald-100 text-emerald-800" : "bg-amber-100 text-amber-800"
          }`}>
            {uploadedCount}/7 Uploaded
          </span>
          <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold ${
            allApproved ? "bg-emerald-100 text-emerald-800" : "bg-slate-100 text-slate-800"
          }`}>
            {approvedCount}/7 Approved
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {batch ? (
            <div className="flex w-64 items-center gap-3 text-sm text-slate-600" role="status">
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-200">
                <div className="h-full bg-blue-600 transition-all" style={{ width: `${(batch.done / batch.total) * 100}%` }} />
              </div>
              <span>{batch.done}/{batch.total} views</span>
            </div>
          ) : (
            pendingProcessing.length > 0 && (
              <button
                onClick={() => runBatch(pendingProcessing)}
                className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800"
              >
                Process {pendingProcessing.length} View{pendingProcessing.length === 1 ? "" : "s"}
              </button>
            )
          )}
          {uploadedCount === 7 && pendingProcessing.length === 0 && drawingCount === 7 && !allApproved && (
            <button
              onClick={async () => {
                for (const v of REQUIRED_VIEW_TYPES) {
                  const img = views[v];
                  if (img && img.approval_status !== "approved") {
                    await reviewSlot(img.id, "approve");
                  }
                }
              }}
              className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500"
            >
              Approve All 7 Views
            </button>
          )}
        </div>
      </div>

      {/* 7 Fixed Upload Slots Grid */}
      <section>
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-slate-900">7 Required Patent Views</h2>
          <p className="text-sm text-slate-500">
            Standard Patent Office representation order: Perspective, Front, Rear, Right Side, Left Side, Top, and Bottom.
          </p>
        </div>

        <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
          {REQUIRED_SLOTS.map((slot) => {
            const img = views[slot.type];
            const working = img ? busy.has(img.id) || isActive(img.processing_status) : false;
            const action = img && !working ? processActionLabel(img.processing_status) : null;
            const hasDrawing = Boolean(img?.result?.urls.drawing);
            const isWarning =
              img?.result?.quality_status === "review_required" ||
              img?.processing_status === "Review required" ||
              Boolean(img?.result?.quality_warnings?.length);

            return (
              <SlotCard
                key={slot.type}
                slot={slot}
                img={img}
                working={working}
                action={action}
                hasDrawing={hasDrawing}
                isWarning={isWarning}
                onUpload={(file) => uploadToSlot(file, slot.type)}
                onProcess={() => img && processImageById(img.id)}
                onReview={(act) => img && reviewSlot(img.id, act)}
                onRemove={() => removeSlot(slot.type)}
              />
            );
          })}
        </div>
      </section>

      {/* Patent Application Document Generation (7-Page PDF) */}
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold tracking-tight text-slate-900">
              Official 7-Page Patent Application PDF
            </h2>
            <p className="mt-1 text-sm text-slate-500 max-w-2xl">
              Generates the publication-ready representation sheet following the exact 7-page order and layout of the Patent Office standard (Perspective, Front, Rear, Right, Left, Top, Bottom).
            </p>
          </div>
          <div>
            <button
              disabled={!allApproved || generatingPdf}
              onClick={handleGeneratePdf}
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {generatingPdf ? (
                <>
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Generating 7-Page PDF…
                </>
              ) : (
                <>
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Generate 7-Page Patent PDF
                </>
              )}
            </button>
          </div>
        </div>

        {/* Validation Checklist Banner */}
        <div className={`mb-6 rounded-xl p-4 text-sm ${
          allApproved ? "bg-emerald-50 text-emerald-900 border border-emerald-200" : "bg-amber-50 text-amber-900 border border-amber-200"
        }`}>
          <div className="flex items-center justify-between">
            <span className="font-semibold">
              {allApproved
                ? "✓ All 7 Views Approved: Document ready for generation."
                : `Document Generation Locked: ${approvedCount} of 7 views approved.`}
            </span>
            <span className="text-xs font-medium">
              {approvedCount}/7 Approved · {uploadedCount}/7 Uploaded
            </span>
          </div>
          {!allApproved && (
            <p className="mt-1 text-xs text-amber-800">
              Each of the 7 slots (Perspective, Front, Rear, Right, Left, Top, Bottom) must have an uploaded photo and an approved technical drawing before generating the final 7-page representation document.
            </p>
          )}
        </div>

        {/* Patent Document Metadata Form */}
        <form onSubmit={saveMetadata} className="space-y-4 rounded-xl border border-slate-100 bg-slate-50/50 p-5">
          <div className="flex items-center justify-between border-b border-slate-200/60 pb-3">
            <h3 className="text-sm font-semibold text-slate-800">Application & Applicant Information</h3>
            <button
              type="submit"
              disabled={savingMeta}
              className="rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
            >
              {savingMeta ? "Saving…" : "Save Details"}
            </button>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <div>
              <label className="block text-xs font-medium text-slate-700">Applicant / Owner Name</label>
              <input
                type="text"
                placeholder="e.g. Acme Innovations Ltd."
                value={metaForm.applicant_name ?? ""}
                onChange={(e) => setMetaForm((m) => ({ ...m, applicant_name: e.target.value }))}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700">Title of Article / Design</label>
              <input
                type="text"
                placeholder="e.g. 2-in-1 Jug"
                value={metaForm.product_title ?? ""}
                onChange={(e) => setMetaForm((m) => ({ ...m, product_title: e.target.value }))}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700">Application Number (Optional)</label>
              <input
                type="text"
                placeholder="e.g. 402911-001 (or leave blank)"
                value={metaForm.application_number ?? ""}
                onChange={(e) => setMetaForm((m) => ({ ...m, application_number: e.target.value }))}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700">Representation / Filing Date</label>
              <input
                type="text"
                placeholder="e.g. 18/11/2026"
                value={metaForm.filing_date ?? ""}
                onChange={(e) => setMetaForm((m) => ({ ...m, filing_date: e.target.value }))}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700">Patent Agent / Attorney Name</label>
              <input
                type="text"
                placeholder="e.g. Jane Doe"
                value={metaForm.agent_name ?? ""}
                onChange={(e) => setMetaForm((m) => ({ ...m, agent_name: e.target.value }))}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700">Agent Registration Number</label>
              <input
                type="text"
                placeholder="e.g. IN/PA-1234"
                value={metaForm.agent_registration_number ?? ""}
                onChange={(e) => setMetaForm((m) => ({ ...m, agent_registration_number: e.target.value }))}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm"
              />
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="block text-xs font-medium text-slate-700">Statement of Novelty</label>
              <textarea
                rows={3}
                value={metaForm.novelty_statement ?? ""}
                onChange={(e) => setMetaForm((m) => ({ ...m, novelty_statement: e.target.value }))}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-mono"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700">Disclaimer</label>
              <textarea
                rows={3}
                value={metaForm.disclaimer ?? ""}
                onChange={(e) => setMetaForm((m) => ({ ...m, disclaimer: e.target.value }))}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-mono"
              />
            </div>
          </div>

          {/* Signature Upload / Preview */}
          <div className="border-t border-slate-200/60 pt-3 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="text-xs font-medium text-slate-700">Agent / Signatory Signature:</span>
              {metadata?.has_signature ? (
                <div className="flex items-center gap-2">
                  <span className="inline-flex items-center rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-800">
                    ✓ Signature on file
                  </span>
                  <button
                    type="button"
                    disabled={sigUploading}
                    onClick={handleSignatureRemove}
                    className="text-xs text-red-600 hover:underline"
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <span className="text-xs text-slate-400">None uploaded (placeholder box will appear in PDF)</span>
              )}
            </div>

            <div>
              <input
                ref={signatureInputRef}
                type="file"
                accept="image/png,image/jpeg"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleSignatureFile(f);
                }}
              />
              <button
                type="button"
                disabled={sigUploading}
                onClick={() => signatureInputRef.current?.click()}
                className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
              >
                {sigUploading ? "Uploading…" : metadata?.has_signature ? "Replace Signature Image" : "Upload Signature Image (PNG/JPG)"}
              </button>
            </div>
          </div>
        </form>

        {/* Generated Documents List */}
        {documents.length > 0 && (
          <div className="mt-6 space-y-3">
            <h3 className="text-sm font-semibold text-slate-800">Generated Patent Representation Documents</h3>
            <ul className="divide-y divide-slate-100 rounded-xl border border-slate-200 bg-white">
              {documents.map((doc) => (
                <li key={doc.id} className="flex flex-wrap items-center justify-between gap-3 p-4">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-red-50 text-red-700">
                      <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-slate-900">{doc.filename}</p>
                      <p className="text-xs text-slate-500">
                        {doc.page_count} Pages (Official Patent Order) · {formatDate(doc.created_at)}
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handlePreview(doc)}
                      className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50"
                    >
                      Preview
                    </button>
                    <button
                      onClick={() => handleDownload(doc)}
                      className="rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-slate-800"
                    >
                      Download PDF
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>
    </div>
  );
}

interface SlotCardProps {
  slot: SlotConfig;
  img: ImageItem | null;
  working: boolean;
  action: string | null;
  hasDrawing: boolean;
  isWarning: boolean;
  onUpload: (file: File) => void;
  onProcess: () => void;
  onReview: (act: "approve" | "reject") => void;
  onRemove: () => void;
}

function SlotCard({
  slot,
  img,
  working,
  action,
  hasDrawing,
  isWarning,
  onUpload,
  onProcess,
  onReview,
  onRemove,
}: SlotCardProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  function handleFiles(files: FileList | null) {
    if (!files?.length) return;
    onUpload(files[0]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        handleFiles(e.dataTransfer.files);
      }}
      className={`relative flex flex-col justify-between overflow-hidden rounded-2xl border bg-white shadow-sm transition ${
        dragOver ? "border-blue-500 bg-blue-50/30" : "border-slate-200"
      }`}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />

      {/* Card Header */}
      <div className="flex items-center justify-between border-b border-slate-100 p-3.5">
        <div className="flex items-center gap-2">
          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-900 text-xs font-semibold text-white">
            {slot.order}
          </span>
          <span className="text-sm font-semibold text-slate-800">{slot.label}</span>
        </div>
        <div className="flex items-center gap-1.5">
          {img ? (
            <>
              <StatusBadge status={working ? "Processing" : img.processing_status} />
              {img.approval_status === "approved" && (
                <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-800 ring-1 ring-emerald-300">
                  ✓ Approved
                </span>
              )}
              {img.approval_status === "rejected" && (
                <span className="rounded-full bg-rose-100 px-2 py-0.5 text-xs font-semibold text-rose-800 ring-1 ring-rose-300">
                  ✕ Rejected
                </span>
              )}
              {img.approval_status === "pending" && hasDrawing && (
                <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-800 ring-1 ring-amber-300">
                  Review
                </span>
              )}
            </>
          ) : (
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-500">
              Empty slot
            </span>
          )}
        </div>
      </div>

      {/* Card Body */}
      {img ? (
        <div className="flex flex-col">
          {/* Side-by-side previews: Photo & Drawing */}
          <Link href={`/images/${img.id}`} className="relative grid grid-cols-2 border-b border-slate-100 hover:opacity-95" aria-label={`Open ${slot.label}`}>
            <div className="relative">
              <span className="absolute left-2 top-2 z-10 rounded bg-slate-900/70 px-1.5 py-0.5 text-[10px] font-medium text-white backdrop-blur-[1px]">
                Photo
              </span>
              <SecureImage
                src={img.original_url}
                alt={`${slot.label} Photo`}
                className="h-36 w-full bg-slate-50 object-contain"
              />
            </div>
            <div className="relative border-l border-slate-100">
              <span className="absolute left-2 top-2 z-10 rounded bg-slate-900/70 px-1.5 py-0.5 text-[10px] font-medium text-white backdrop-blur-[1px]">
                Drawing
              </span>
              <SecureImage
                src={img.result?.urls.drawing ?? null}
                alt={`${slot.label} Technical Drawing`}
                placeholder={img.processing_status === "Failed" ? "Failed" : "Not generated"}
                className="h-36 w-full bg-white object-contain p-2"
                version={img.result?.updated_at}
              />
            </div>
            {working && (
              <div className="absolute inset-0 flex items-center justify-center gap-2 bg-white/80 text-sm font-medium text-slate-700 backdrop-blur-[1px]">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-blue-600" />
                Generating drawing…
              </div>
            )}
          </Link>

          <div className="space-y-2.5 p-3.5">
            <div className="flex items-center justify-between text-xs text-slate-500">
              <span className="truncate max-w-[170px]" title={img.original_filename}>{img.original_filename}</span>
              <span>{img.width}×{img.height} · {formatBytes(img.size_bytes)}</span>
            </div>

            {/* Quality validation warnings */}
            {isWarning && img.result?.quality_warnings && img.result.quality_warnings.length > 0 && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 p-2.5 text-xs text-amber-900">
                <p className="font-semibold">⚠️ Quality Validation Notice:</p>
                <ul className="mt-1 list-inside list-disc space-y-0.5 text-[11px] text-amber-800">
                  {img.result.quality_warnings.map((w, idx) => (
                    <li key={idx}>{w}</li>
                  ))}
                </ul>
              </div>
            )}

            {img.processing_status === "Failed" && img.result?.error_message && (
              <p className="rounded-lg bg-red-50 p-2 text-xs text-red-800">
                <strong>{stageLabel(img.result.failed_stage)}:</strong> {img.result.error_message}
              </p>
            )}

            {/* Action buttons */}
            <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
              <div className="flex gap-1.5">
                {hasDrawing && img.approval_status !== "approved" && (
                  <button
                    onClick={() => onReview("approve")}
                    className="rounded-lg bg-emerald-600 px-2.5 py-1 text-xs font-semibold text-white shadow-sm hover:bg-emerald-500"
                  >
                    Approve
                  </button>
                )}
                {hasDrawing && img.approval_status !== "rejected" && (
                  <button
                    onClick={() => onReview("reject")}
                    className="rounded-lg border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
                  >
                    Reject
                  </button>
                )}
                {action && (
                  <button
                    onClick={onProcess}
                    className="rounded-lg bg-slate-900 px-2.5 py-1 text-xs font-semibold text-white hover:bg-slate-800"
                  >
                    {action}
                  </button>
                )}
              </div>

              <div className="flex items-center gap-1.5">
                {hasDrawing && (
                  <Link
                    href={`/images/${img.id}`}
                    className="rounded-lg border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-blue-600 hover:bg-blue-50"
                    title="Inspect 8 pipeline stages"
                  >
                    Diagnostics
                  </Link>
                )}
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="rounded-lg border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
                  title="Replace photo with another file"
                >
                  Replace
                </button>
                <button
                  type="button"
                  onClick={onRemove}
                  className="rounded-lg border border-slate-300 px-2 py-1 text-xs text-slate-400 hover:text-red-700 hover:border-red-200"
                  title="Remove from this slot"
                >
                  ✕
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* Empty Slot State */
        <div
          role="button"
          tabIndex={0}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && fileInputRef.current?.click()}
          className="flex flex-1 cursor-pointer flex-col items-center justify-center p-8 text-center hover:bg-slate-50/70"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-100 text-slate-400">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75" d="M12 4v16m8-8H4" />
            </svg>
          </div>
          <p className="mt-2 text-xs font-semibold text-slate-800">Upload {slot.label}</p>
          <p className="mt-0.5 text-[11px] text-slate-400 max-w-[200px]">{slot.description}</p>
          <span className="mt-2 text-[11px] font-medium text-blue-600 underline">Drop photo or browse</span>
        </div>
      )}
    </div>
  );
}

export default function ProjectPage() {
  const params = useParams<{ id: string }>();
  return (
    <AppShell>
      <ProjectView id={Number(params.id)} />
    </AppShell>
  );
}