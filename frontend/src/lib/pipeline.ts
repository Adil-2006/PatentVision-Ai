import type { ImageStatus } from "./types";

export const STAGES = [
  { key: "load", label: "Load photo" },
  { key: "background_removal", label: "Remove background" },
  { key: "localization", label: "Locate product" },
  { key: "segmentation", label: "Segment" },
  { key: "mask_validation", label: "Validate mask" },
  { key: "contour_extraction", label: "Trace outline" },
  { key: "rendering", label: "Render drawing" },
  { key: "output_validation", label: "Check output" },
] as const;

export const WORKFLOW_STEPS = [
  "Image uploaded",
  "Removing background",
  "Background removal completed",
  "Extracting structural details",
  "Generating technical drawing",
  "Validating drawing quality",
  "Ready for review",
] as const;

export type StageState = "done" | "failed" | "active" | "pending";

/** Per-stage state for the pipeline stepper, derived from the job status and the failed stage. */
export function stageStates(status: ImageStatus, failedStage: string | null): StageState[] {
  const n = STAGES.length;
  switch (status) {
    case "Completed":
    case "Review required":
      return Array(n).fill("done");
    case "Queued":
    case "Processing":
      return Array(n).fill("active");
    case "Failed": {
      const idx = STAGES.findIndex((s) => s.key === failedStage);
      if (idx === -1) {
        // save / persistence failures happen after every image stage succeeded; interruptions are unknown
        const after = failedStage === "save" || failedStage === "persistence";
        return Array(n).fill(after ? "done" : "pending");
      }
      return STAGES.map((_, i) => (i < idx ? "done" : i === idx ? "failed" : "pending"));
    }
    default:
      return Array(n).fill("pending");
  }
}

/** Return the active 0-based milestone index across the 7 user-facing workflow steps. */
export function workflowStepIndex(status: ImageStatus, hasBgRemoved: boolean): number {
  switch (status) {
    case "Completed":
    case "Review required":
      return WORKFLOW_STEPS.length - 1;
    case "Queued":
      return 1;
    case "Processing":
      return hasBgRemoved ? 3 : 1;
    default:
      return 0;
  }
}
