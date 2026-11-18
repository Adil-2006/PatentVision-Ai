import { describe, expect, it } from "vitest";
import { REQUIRED_VIEW_TYPES, type ImageItem } from "./types";
import {
  detectViewTypeFromFilename,
  REQUIRED_SLOTS,
  validateProjectViewsReadiness,
} from "./views";

describe("detectViewTypeFromFilename", () => {
  it("detects perspective view filenames", () => {
    expect(detectViewTypeFromFilename("jug_perspective.jpg")).toBe("Perspective");
    expect(detectViewTypeFromFilename("product_iso_angle.png")).toBe("Perspective");
    expect(detectViewTypeFromFilename("3d_render.webp")).toBe("Perspective");
  });

  it("detects front view filenames", () => {
    expect(detectViewTypeFromFilename("DSC_001_front.jpg")).toBe("Front");
    expect(detectViewTypeFromFilename("front_view.png")).toBe("Front");
  });

  it("detects rear and back filenames", () => {
    expect(detectViewTypeFromFilename("jug_rear.jpg")).toBe("Rear");
    expect(detectViewTypeFromFilename("back_elevation.png")).toBe("Rear");
  });

  it("detects right and left filenames", () => {
    expect(detectViewTypeFromFilename("right_side.jpg")).toBe("Right");
    expect(detectViewTypeFromFilename("left_side.jpg")).toBe("Left");
  });

  it("detects top and bottom filenames", () => {
    expect(detectViewTypeFromFilename("top_plan.jpg")).toBe("Top");
    expect(detectViewTypeFromFilename("overhead.png")).toBe("Top");
    expect(detectViewTypeFromFilename("bottom_base.jpg")).toBe("Bottom");
    expect(detectViewTypeFromFilename("underside.webp")).toBe("Bottom");
  });

  it("returns null for generic filenames", () => {
    expect(detectViewTypeFromFilename("photo_1.jpg")).toBeNull();
    expect(detectViewTypeFromFilename("random_name.png")).toBeNull();
  });
});

describe("REQUIRED_SLOTS order and structure", () => {
  it("preserves exactly 7 slots in official patent order", () => {
    expect(REQUIRED_SLOTS).toHaveLength(7);
    const order = REQUIRED_SLOTS.map((s) => s.type);
    expect(order).toEqual([
      "Perspective",
      "Front",
      "Rear",
      "Right",
      "Left",
      "Top",
      "Bottom",
    ]);
  });
});

describe("validateProjectViewsReadiness", () => {
  function makeMockImage(
    view: ImageItem["view_type"],
    hasDrawing: boolean,
    approval: ImageItem["approval_status"]
  ): ImageItem {
    return {
      id: 1,
      project_id: 1,
      original_filename: `${view.toLowerCase()}.jpg`,
      mime_type: "image/jpeg",
      size_bytes: 1024,
      width: 800,
      height: 600,
      view_type: view,
      processing_status: hasDrawing ? "Completed" : "Uploaded",
      approval_status: approval,
      is_active: true,
      created_at: "2026-11-18T10:00:00Z",
      original_url: "/url",
      result: hasDrawing
        ? {
            id: 1,
            image_id: 1,
            processing_time_ms: 500,
            error_message: null,
            failed_stage: null,
            urls: {
              mask: "/m",
              drawing: "/d",
              transparent: "/t",
              vector_svg: "/s",
            },
            stages: [],
            metrics: {},
            quality_status: "passed",
            quality_warnings: [],
            updated_at: "2026-11-18T10:00:00Z",
          }
        : null,
      job: null,
    };
  }

  it("identifies missing views when some slots are empty", () => {
    const emptyViews = {
      Perspective: makeMockImage("Perspective", true, "approved"),
      Front: null,
      Rear: null,
      Right: null,
      Left: null,
      Top: null,
      Bottom: null,
    };
    const res = validateProjectViewsReadiness(emptyViews);
    expect(res.isReadyForDocument).toBe(false);
    expect(res.uploadedCount).toBe(1);
    expect(res.approvedCount).toBe(1);
    expect(res.missingViews).toContain("Front");
    expect(res.missingViews).toContain("Rear");
  });

  it("requires approval even if all drawings are completed", () => {
    const unapprovedViews: Record<string, ImageItem | null> = {};
    for (const v of REQUIRED_VIEW_TYPES) {
      unapprovedViews[v] = makeMockImage(v, true, "pending");
    }
    const res = validateProjectViewsReadiness(unapprovedViews as any);
    expect(res.isReadyForDocument).toBe(false);
    expect(res.uploadedCount).toBe(7);
    expect(res.drawingCount).toBe(7);
    expect(res.approvedCount).toBe(0);
    expect(res.unapprovedViews).toHaveLength(7);
  });

  it("marks ready for document only when all 7 views are approved", () => {
    const readyViews: Record<string, ImageItem | null> = {};
    for (const v of REQUIRED_VIEW_TYPES) {
      readyViews[v] = makeMockImage(v, true, "approved");
    }
    const res = validateProjectViewsReadiness(readyViews as any);
    expect(res.isReadyForDocument).toBe(true);
    expect(res.uploadedCount).toBe(7);
    expect(res.drawingCount).toBe(7);
    expect(res.approvedCount).toBe(7);
    expect(res.missingViews).toHaveLength(0);
    expect(res.unapprovedViews).toHaveLength(0);
  });
});
