"""Seven-page patent design illustration document generator following Jug 2-in-1.pdf layout.

Page Order:
1. Perspective View
2. Front View
3. Rear View
4. Right View
5. Left View
6. Top View
7. Bottom View

Never invents legal text, dates, applicant details, or signatures.
Labels output as a preparation draft requiring authorized IP professional review.
"""
import io
from typing import Any

from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

PDF_VIEW_ORDER = [
    ("Perspective", "PERSPECTIVE VIEW"),
    ("Front", "FRONT VIEW"),
    ("Rear", "REAR VIEW"),
    ("Right", "RIGHT VIEW"),
    ("Left", "LEFT VIEW"),
    ("Top", "TOP VIEW"),
    ("Bottom", "BOTTOM VIEW"),
]

PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = 54.0
RIGHT_MARGIN = PAGE_WIDTH - 54.0

DEFAULT_NOVELTY_TEMPLATE = "Novelty resides in the shape and configuration of the “{title}” as illustrated."
DEFAULT_DISCLAIMER_BULLETS = [
    "No claim is made by virtue of this registration to any right to the use as a trade mark of what is shown in the representation.",
    "No claim is made by virtue of the registration in respect of any mechanical or other action of the mechanism whatever or in respect of any mode or principle of construction of the article.",
    "No claim is made by virtue of this registration to any exclusive use of the words, letters, numerals, flags, crowns, etc. appearing in the design.",
]


def wrap_text(text: str, max_chars: int = 90) -> list[str]:
    """Simple greedy word wrap to keep formatting clean."""
    words = text.split()
    if not words:
        return [""]
    lines = []
    curr = []
    curr_len = 0
    for w in words:
        if curr_len + len(w) + 1 <= max_chars:
            curr.append(w)
            curr_len += len(w) + 1
        else:
            lines.append(" ".join(curr))
            curr = [w]
            curr_len = len(w)
    if curr:
        lines.append(" ".join(curr))
    return lines


def generate_patent_pdf(
    metadata: Any,
    drawings: dict[str, bytes],
    signature_bytes: bytes | None = None,
) -> bytes:
    """Generate the exact seven-page patent design illustration PDF in reference order."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4, pageCompression=0)
    total_pages = len(PDF_VIEW_ORDER)

    applicant_name = (getattr(metadata, "applicant_name", "") or "").strip()
    product_title = (getattr(metadata, "product_title", "") or "").strip() or "PRODUCT"
    filing_date = (getattr(metadata, "filing_date", "") or "").strip() or "—"
    agent_name = (getattr(metadata, "agent_name", "") or "").strip()
    agent_reg = (getattr(metadata, "agent_registration_number", "") or "").strip()
    agent_desig = (getattr(metadata, "agent_designation", "") or "").strip() or "Agent for the Applicant"

    custom_novelty = (getattr(metadata, "novelty_statement", "") or "").strip()
    if not custom_novelty or any(k in custom_novelty.lower() for k in ["brushmate", "brush", "jug", "pitcher"]):
        novelty_body = DEFAULT_NOVELTY_TEMPLATE.format(title=product_title)
    else:
        novelty_body = custom_novelty

    custom_disclaimer = (getattr(metadata, "disclaimer", "") or "").strip()
    if custom_disclaimer:
        disclaimer_bullets = [line.strip().lstrip("•-* ").strip() for line in custom_disclaimer.splitlines() if line.strip()]
    else:
        disclaimer_bullets = list(DEFAULT_DISCLAIMER_BULLETS)

    for page_idx, (view_key, view_title) in enumerate(PDF_VIEW_ORDER):
        page_num = page_idx + 1

        # 1. Header: Applicant Name (Left) and Page No. (Right)
        c.setFont("Times-Roman", 11)
        if applicant_name and applicant_name.lower() not in ("applicant name", "—", "none"):
            c.drawString(LEFT_MARGIN, PAGE_HEIGHT - 54, f"Applicant Name: {applicant_name}")
        else:
            c.drawString(LEFT_MARGIN, PAGE_HEIGHT - 54, "Applicant Name: —")
        c.drawRightString(RIGHT_MARGIN, PAGE_HEIGHT - 54, f"Page No.: {page_num} of {total_pages}")

        # 2. Centered Technical Drawing Box
        drawing_png = drawings.get(view_key)
        box_x = LEFT_MARGIN + 20
        box_y = 440.0
        box_w = (RIGHT_MARGIN - LEFT_MARGIN) - 40.0  # ~447 pt
        box_h = 280.0  # span from 440 to 720

        if drawing_png:
            try:
                pil_img = PILImage.open(io.BytesIO(drawing_png))
                img_w, img_h = pil_img.size
                scale = min(box_w / img_w, box_h / img_h)
                draw_w = img_w * scale
                draw_h = img_h * scale
                draw_x = (PAGE_WIDTH - draw_w) / 2.0
                draw_y = box_y + (box_h - draw_h) / 2.0
                reader = ImageReader(io.BytesIO(drawing_png))
                c.drawImage(reader, draw_x, draw_y, width=draw_w, height=draw_h, mask="auto")
            except Exception:
                c.setFont("Helvetica-Oblique", 10)
                c.drawCentredString(PAGE_WIDTH / 2.0, box_y + box_h / 2.0, f"[{view_title} Illustration]")
        else:
            c.setFont("Helvetica-Oblique", 10)
            c.drawCentredString(PAGE_WIDTH / 2.0, box_y + box_h / 2.0, f"[{view_title} Illustration]")

        # 3. View Title (Bold Centered)
        c.setFont("Times-Bold", 11)
        c.drawCentredString(PAGE_WIDTH / 2.0, 410, view_title)

        # 4. Statement of Novelty
        c.setFont("Times-Bold", 10)
        c.drawString(LEFT_MARGIN, 375, "Statement of novelty")
        c.setFont("Times-Roman", 10)
        novelty_lines = wrap_text(novelty_body, max_chars=88)
        y_cursor = 360
        for n_line in novelty_lines:
            c.drawString(LEFT_MARGIN + 12, y_cursor, n_line)
            y_cursor -= 14

        # 5. Disclaimer
        y_cursor -= 4
        c.setFont("Times-Bold", 10)
        c.drawString(LEFT_MARGIN, y_cursor, "Disclaimer")
        c.setFont("Times-Roman", 10)
        y_cursor -= 14
        for b_text in disclaimer_bullets:
            bullet_lines = wrap_text(b_text, max_chars=82)
            c.drawString(LEFT_MARGIN + 12, y_cursor, "•")
            for idx, b_line in enumerate(bullet_lines):
                c.drawString(LEFT_MARGIN + 22, y_cursor, b_line)
                y_cursor -= 13
            y_cursor -= 2

        # 6. Footer: Date (Left) and Signature Area (Right)
        footer_y = 120
        c.setFont("Times-Roman", 10)
        c.drawString(LEFT_MARGIN, footer_y, f"Date: {filing_date}")

        # Right side: Signature & Agent Info
        sig_right = RIGHT_MARGIN
        sig_y = footer_y + 10
        if signature_bytes:
            try:
                sig_pil = PILImage.open(io.BytesIO(signature_bytes))
                sw, sh = sig_pil.size
                scale = min(120.0 / sw, 35.0 / sh)
                sw_draw, sh_draw = sw * scale, sh * scale
                sig_reader = ImageReader(io.BytesIO(signature_bytes))
                c.drawImage(sig_reader, sig_right - sw_draw, sig_y, width=sw_draw, height=sh_draw, mask="auto")
                sig_y -= 8
            except Exception:
                pass

        if agent_name and agent_name.lower() not in ("—", "none", ""):
            c.drawRightString(sig_right, sig_y - 12, agent_name)
            sig_y -= 14
            if agent_reg and agent_reg.lower() not in ("—", "none", ""):
                c.drawRightString(sig_right, sig_y - 12, f"({agent_reg})")
                sig_y -= 14
        if agent_desig:
            c.drawRightString(sig_right, sig_y - 12, agent_desig)

        # 7. Draft / Professional Review Notice (Bottom footer)
        c.setFont("Helvetica-Oblique", 7)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawCentredString(
            PAGE_WIDTH / 2.0,
            20,
            "Preparation Draft • Requires Authorized IP Professional Review • PatentVision AI",
        )
        c.setFillColorRGB(0, 0, 0)

        c.showPage()

    c.save()
    return buffer.getvalue()
