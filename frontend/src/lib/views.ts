import {
  REQUIRED_VIEW_TYPES,
  type ImageItem,
  type RequiredViewType,
} from "./types";

export interface SlotConfig {
  type: RequiredViewType;
  label: string;
  order: number;
  description: string;
}

export const REQUIRED_SLOTS: SlotConfig[] = [
  {
    type: "Perspective",
    label: "Perspective View",
    order: 1,
    description: "3D perspective / exploded view of complete product",
  },
  {
    type: "Front",
    label: "Front View",
    order: 2,
    description: "Direct front elevation representation",
  },
  {
    type: "Rear",
    label: "Rear View",
    order: 3,
    description: "Direct rear elevation representation",
  },
  {
    type: "Right",
    label: "Right Side View",
    order: 4,
    description: "Direct right profile representation",
  },
  {
    type: "Left",
    label: "Left Side View",
    order: 5,
    description: "Direct left profile representation",
  },
  {
    type: "Top",
    label: "Top View",
    order: 6,
    description: "Direct top / plan elevation representation",
  },
  {
    type: "Bottom",
    label: "Bottom View",
    order: 7,
    description: "Direct bottom / underside elevation representation",
  },
];

/** Heuristic to detect standard patent view slot from filename */
export function detectViewTypeFromFilename(filename: string): RequiredViewType | null {
  const name = filename.toLowerCase();
  if (/persp|iso|3d|angle|slant|overall/i.test(name)) return "Perspective";
  if (/front/i.test(name)) return "Front";
  if (/rear|back/i.test(name)) return "Rear";
  if (/right/i.test(name)) return "Right";
  if (/left/i.test(name)) return "Left";
  if (/top|overhead|plan/i.test(name)) return "Top";
  if (/bot|bottom|base|under/i.test(name)) return "Bottom";
  return null;
}

/** Check if all 7 required view slots are uploaded and approved */
export function validateProjectViewsReadiness(
  views: Record<RequiredViewType, ImageItem | null>
): {
  isReadyForDocument: boolean;
  uploadedCount: number;
  drawingCount: number;
  approvedCount: number;
  missingViews: RequiredViewType[];
  unapprovedViews: RequiredViewType[];
} {
  const missingViews: RequiredViewType[] = [];
  const unapprovedViews: RequiredViewType[] = [];
  let uploadedCount = 0;
  let drawingCount = 0;
  let approvedCount = 0;

  for (const v of REQUIRED_VIEW_TYPES) {
    const img = views[v];
    if (!img) {
      missingViews.push(v);
      unapprovedViews.push(v);
    } else {
      uploadedCount++;
      if (img.result?.urls.drawing) {
        drawingCount++;
      }
      if (img.approval_status === "approved") {
        approvedCount++;
      } else {
        unapprovedViews.push(v);
      }
    }
  }

  return {
    isReadyForDocument: approvedCount === REQUIRED_VIEW_TYPES.length,
    uploadedCount,
    drawingCount,
    approvedCount,
    missingViews,
    unapprovedViews,
  };
}
