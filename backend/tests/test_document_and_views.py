import io
import cv2
import numpy as np
import pytest

from sqlalchemy import select

from app.core import database
from app.core.storage import storage
from app.models import REQUIRED_VIEW_TYPES, AIResult, DocumentMetadata, Image, Project, ProjectDocument
from app.services.document_generation import PDF_VIEW_ORDER, generate_patent_pdf
from tests import synthetic
from tests.conftest import register, upload


def sample_png():
    img = np.full((300, 300, 3), 240, dtype=np.uint8)
    cv2.circle(img, (150, 150), 80, (40, 40, 40), -1)
    return synthetic.encode(img)


def test_perspective_is_valid_view_and_accepted(client, auth, project):
    pid = project["id"]
    r = upload(client, auth, pid, sample_png(), "persp.png", view="Perspective")
    assert r.status_code == 201
    body = r.json()
    assert body["view_type"] == "Perspective"
    assert body["approval_status"] == "pending"
    assert body["is_active"] is True


def test_upload_or_replace_view_endpoint(client, auth, project):
    pid = project["id"]
    png1 = sample_png()
    r1 = client.post(
        f"/projects/{pid}/views/Front",
        files={"file": ("front1.png", png1, "image/png")},
        headers=auth,
    )
    assert r1.status_code == 201
    img1_id = r1.json()["id"]

    # Invalidate by replacing with a second image in Front slot
    png2 = sample_png()
    r2 = client.post(
        f"/projects/{pid}/views/Front",
        files={"file": ("front2.png", png2, "image/png")},
        headers=auth,
    )
    assert r2.status_code == 201
    assert r2.json()["original_filename"] == "front2.png"

    # Check that old image record was removed and only 1 Front view is active
    with database.SessionLocal() as db:
        images = db.scalars(
            select(Image).where(Image.project_id == pid, Image.view_type == "Front")
        ).all()
        assert len(images) == 1
        assert images[0].original_filename == "front2.png"


def test_list_project_views_status_and_readiness(client, auth, project):
    pid = project["id"]
    # Initially all 7 slots missing
    r = client.get(f"/projects/{pid}/views", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["is_ready_to_process"] is False
    assert body["is_ready_for_document"] is False
    assert len(body["missing_views"]) == 7
    assert set(body["view_order"]) == set(REQUIRED_VIEW_TYPES)

    # Upload all 7 required views
    for v in REQUIRED_VIEW_TYPES:
        r_up = client.post(
            f"/projects/{pid}/views/{v}",
            files={"file": (f"{v.lower()}.png", sample_png(), "image/png")},
            headers=auth,
        )
        assert r_up.status_code == 201

    # Now ready to process, but not ready for document (unapproved)
    r2 = client.get(f"/projects/{pid}/views", headers=auth)
    body2 = r2.json()
    assert body2["is_ready_to_process"] is True
    assert len(body2["missing_views"]) == 0
    assert body2["is_ready_for_document"] is False
    assert len(body2["unapproved_views"]) == 7


def test_review_approval_and_rejection_workflow(client, auth, project):
    pid = project["id"]
    r_up = client.post(
        f"/projects/{pid}/views/Top",
        files={"file": ("top.png", sample_png(), "image/png")},
        headers=auth,
    )
    img_id = r_up.json()["id"]

    # Cannot approve before drawing is generated
    r_bad = client.post(f"/images/{img_id}/review", json={"action": "approve"}, headers=auth)
    assert r_bad.status_code == 400
    assert r_bad.json()["error"]["code"] == "cannot_approve"

    # Process the image
    r_proc = client.post(f"/ai/process/{img_id}", headers=auth)
    assert r_proc.status_code == 200

    # Reject
    r_rej = client.post(f"/images/{img_id}/review", json={"action": "reject"}, headers=auth)
    assert r_rej.status_code == 200
    assert r_rej.json()["approval_status"] == "rejected"

    # Approve
    r_app = client.post(f"/images/{img_id}/review", json={"action": "approve"}, headers=auth)
    assert r_app.status_code == 200
    assert r_app.json()["approval_status"] == "approved"
    assert r_app.json()["processing_status"] == "Completed"


def test_document_metadata_crud_and_signature(client, auth, project):
    pid = project["id"]
    # Get default metadata
    r = client.get(f"/projects/{pid}/document-metadata", headers=auth)
    assert r.status_code == 200
    meta = r.json()
    assert meta["product_title"] == project["name"]
    assert meta["has_signature"] is False

    # Update metadata
    r_up = client.put(
        f"/projects/{pid}/document-metadata",
        json={
            "applicant_name": "RV University",
            "product_title": "BRUSHMATE-PRO",
            "application_number": "APP-12345",
            "filing_date": "01/07/2025",
            "novelty_statement": "Novelty resides in the shape and configuration of the “BRUSHMATE-PRO” as illustrated.",
            "disclaimer": "• No claim 1\n• No claim 2\n• No claim 3",
            "agent_name": "Phani Kumar Pullela",
            "agent_registration_number": "IN/PA-4726",
            "agent_designation": "Agent for the Applicant",
        },
        headers=auth,
    )
    assert r_up.status_code == 200
    updated = r_up.json()
    assert updated["applicant_name"] == "RV University"
    assert updated["product_title"] == "BRUSHMATE-PRO"
    assert updated["agent_name"] == "Phani Kumar Pullela"

    # Upload signature
    sig_data = sample_png()
    r_sig = client.post(
        f"/projects/{pid}/document-metadata/signature",
        files={"file": ("signature.png", sig_data, "image/png")},
        headers=auth,
    )
    assert r_sig.status_code == 200
    assert r_sig.json()["has_signature"] is True

    # Retrieve signature image
    r_sig_img = client.get(f"/projects/{pid}/document-metadata/signature", headers=auth)
    assert r_sig_img.status_code == 200
    assert r_sig_img.content == sig_data

    # Delete signature
    r_del = client.delete(f"/projects/{pid}/document-metadata/signature", headers=auth)
    assert r_del.status_code == 204


def test_pdf_generation_blocks_unapproved_or_missing_views(client, auth, project):
    pid = project["id"]
    # Attempt generation with 0 views
    r_fail1 = client.post(f"/projects/{pid}/documents", headers=auth)
    assert r_fail1.status_code == 400
    assert r_fail1.json()["error"]["code"] == "missing_required_views"

    # Upload all 7 views
    imgs = []
    for v in REQUIRED_VIEW_TYPES:
        r = client.post(
            f"/projects/{pid}/views/{v}",
            files={"file": (f"{v.lower()}.png", sample_png(), "image/png")},
            headers=auth,
        )
        imgs.append(r.json()["id"])

    # Attempt generation before processing/approval
    r_fail2 = client.post(f"/projects/{pid}/documents", headers=auth)
    assert r_fail2.status_code == 400
    assert r_fail2.json()["error"]["code"] == "views_not_approved"


def test_pdf_generation_complete_workflow(client, auth, project):
    pid = project["id"]

    # 1. Update metadata
    client.put(
        f"/projects/{pid}/document-metadata",
        json={
            "applicant_name": "RV University",
            "product_title": "BRUSHMATE-PRO",
            "filing_date": "01/07/2025",
            "novelty_statement": "Novelty resides in the shape and configuration of the “BRUSHMATE-PRO” as illustrated.",
            "disclaimer": "• Standard disclaimer 1\n• Standard disclaimer 2\n• Standard disclaimer 3",
            "agent_name": "Phani Kumar Pullela",
            "agent_registration_number": "IN/PA-4726",
            "agent_designation": "Agent for the Applicant",
        },
        headers=auth,
    )

    # 2. Upload, process, and approve all 7 views
    for v in REQUIRED_VIEW_TYPES:
        r_up = client.post(
            f"/projects/{pid}/views/{v}",
            files={"file": (f"{v.lower()}.png", sample_png(), "image/png")},
            headers=auth,
        )
        iid = r_up.json()["id"]
        # Process view
        client.post(f"/ai/process/{iid}", headers=auth)
        # Approve view
        client.post(f"/images/{iid}/review", json={"action": "approve"}, headers=auth)

    # 3. Generate 7-page PDF
    r_gen = client.post(f"/projects/{pid}/documents", headers=auth)
    assert r_gen.status_code == 200
    doc_info = r_gen.json()["document"]
    assert doc_info["page_count"] == 7
    assert doc_info["version"] == 1
    doc_id = doc_info["id"]

    # 4. Download generated PDF
    r_dl = client.get(f"/documents/{doc_id}/download", headers=auth)
    assert r_dl.status_code == 200
    assert r_dl.headers["content-type"] == "application/pdf"
    pdf_bytes = r_dl.content
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 5000

    # 5. Check PDF structure (7 pages in exact order)
    # Search for all 7 view titles in exact order inside the PDF
    titles = [
        b"PERSPECTIVE VIEW",
        b"FRONT VIEW",
        b"REAR VIEW",
        b"RIGHT VIEW",
        b"LEFT VIEW",
        b"TOP VIEW",
        b"BOTTOM VIEW",
    ]
    for t in titles:
        assert t in pdf_bytes, f"Expected {t.decode()} in generated PDF"

    # Verify order of appearance in the PDF stream
    positions = [pdf_bytes.find(t) for t in titles]
    assert positions == sorted(positions), "Views must appear in exact specified order!"

    # Verify applicant name and agent info appear
    assert b"RV University" in pdf_bytes
    assert b"Phani Kumar Pullela" in pdf_bytes
    assert b"IN/PA-4726" in pdf_bytes
    assert b"Preparation Draft" in pdf_bytes

    # 6. Verify version increment
    r_gen2 = client.post(f"/projects/{pid}/documents", headers=auth)
    assert r_gen2.status_code == 200
    assert r_gen2.json()["document"]["version"] == 2
