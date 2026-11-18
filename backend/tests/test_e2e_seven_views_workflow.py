from pathlib import Path
from fastapi.testclient import TestClient

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_views"


def test_complete_e2e_seven_views_patent_workflow(client: TestClient, auth: dict):
    # 1. Create project
    res = client.post(
        "/projects",
        json={"name": "2-in-1 Jug Patent Project", "description": "Patent application for Jug"},
        headers=auth,
    )
    assert res.status_code == 201, res.text
    project = res.json()
    project_id = project["id"]

    # 2. Upload all 7 required views using real product photographs
    view_files = [
        ("Perspective", "perspective.jpg"),
        ("Front", "front.jpg"),
        ("Rear", "rear.jpg"),
        ("Right", "right.jpg"),
        ("Left", "left.jpg"),
        ("Top", "top.jpg"),
        ("Bottom", "bottom.jpg"),
    ]

    image_ids = {}
    for view_type, filename in view_files:
        path = FIXTURES_DIR / filename
        assert path.exists(), f"Fixture {filename} missing"
        with open(path, "rb") as f:
            upload_res = client.post(
                f"/projects/{project_id}/views/{view_type}",
                files={"file": (filename, f, "image/jpeg")},
                headers=auth,
            )
        assert upload_res.status_code == 201, upload_res.text
        img_data = upload_res.json()
        assert img_data["view_type"] == view_type
        assert img_data["is_active"] is True
        image_ids[view_type] = img_data["id"]

    assert len(image_ids) == 7

    # 3. Verify views status endpoint
    views_res = client.get(f"/projects/{project_id}/views", headers=auth)
    assert views_res.status_code == 200
    v_data = views_res.json()
    assert v_data["is_ready_to_process"] is True
    assert v_data["is_ready_for_document"] is False
    assert len(v_data["missing_views"]) == 0
    assert len(v_data["unapproved_views"]) == 7

    # 4. Attempting to generate document before drawings and approvals must fail
    doc_fail = client.post(f"/projects/{project_id}/documents", headers=auth)
    assert doc_fail.status_code == 400
    assert "approved" in doc_fail.json()["error"]["message"].lower()

    # 5. Process all 7 views through real image-processing pipeline
    for view_type, img_id in image_ids.items():
        proc_res = client.post(f"/ai/process/{img_id}", headers=auth)
        assert proc_res.status_code == 200, f"Processing failed for {view_type}: {proc_res.text}"
        data = proc_res.json()
        assert data["status"] in ("Completed", "Review required")

        img_detail = client.get(f"/images/{img_id}", headers=auth).json()
        assert img_detail["result"]["urls"]["drawing"] is not None

    # 6. Still unapproved -> document generation must fail
    doc_fail2 = client.post(f"/projects/{project_id}/documents", headers=auth)
    assert doc_fail2.status_code == 400

    # 7. Review and approve each drawing
    for view_type, img_id in image_ids.items():
        review_res = client.post(
            f"/images/{img_id}/review",
            json={"action": "approve", "notes": f"Approved {view_type} technical drawing"},
            headers=auth,
        )
        assert review_res.status_code == 200
        assert review_res.json()["approval_status"] == "approved"

    # 8. Check views endpoint again: now ready for document!
    views_res2 = client.get(f"/projects/{project_id}/views", headers=auth)
    assert views_res2.status_code == 200
    v_data2 = views_res2.json()
    assert v_data2["is_ready_for_document"] is True
    assert len(v_data2["unapproved_views"]) == 0

    # 9. Update document metadata
    meta_payload = {
        "applicant_name": "Tupperware Products S.A.",
        "product_title": "2-in-1 Jug",
        "application_number": "402911-001",
        "filing_date": "18/11/2026",
        "agent_name": "Patent & Legal Associates",
        "agent_registration_number": "IN/PA-9988",
        "agent_designation": "Agent for the Applicant",
        "novelty_statement": "The novelty resides in the shape and configuration of the 2-in-1 Jug as illustrated.",
        "disclaimer": "No claim is made by virtue of this registration in respect of any mechanical or other action.",
    }
    meta_res = client.put(
        f"/projects/{project_id}/document-metadata",
        json=meta_payload,
        headers=auth,
    )
    assert meta_res.status_code == 200
    meta_out = meta_res.json()
    assert meta_out["applicant_name"] == "Tupperware Products S.A."
    assert meta_out["product_title"] == "2-in-1 Jug"

    # 10. Generate the official 7-page PDF
    gen_res = client.post(f"/projects/{project_id}/documents", headers=auth)
    assert gen_res.status_code == 200, gen_res.text
    doc_info = gen_res.json()["document"]
    assert doc_info["page_count"] == 7
    assert doc_info["filename"].endswith(".pdf")
    doc_id = doc_info["id"]

    # 11. List documents
    list_res = client.get(f"/projects/{project_id}/documents", headers=auth)
    assert list_res.status_code == 200
    docs = list_res.json()
    assert len(docs) == 1
    assert docs[0]["id"] == doc_id
    assert docs[0]["page_count"] == 7

    # 12. Preview and download document
    preview_res = client.get(f"/documents/{doc_id}/preview", headers=auth)
    assert preview_res.status_code == 200
    assert preview_res.headers["content-type"] == "application/pdf"
    pdf_bytes = preview_res.content
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 20000

    # Verify content in generated PDF
    pdf_text = pdf_bytes.decode("latin1", errors="ignore")
    assert "Page No.: 1 of 7" in pdf_text
    assert "Page No.: 2 of 7" in pdf_text
    assert "Page No.: 3 of 7" in pdf_text
    assert "Page No.: 4 of 7" in pdf_text
    assert "Page No.: 5 of 7" in pdf_text
    assert "Page No.: 6 of 7" in pdf_text
    assert "Page No.: 7 of 7" in pdf_text

    assert "PERSPECTIVE VIEW" in pdf_text
    assert "FRONT VIEW" in pdf_text
    assert "REAR VIEW" in pdf_text
    assert "RIGHT VIEW" in pdf_text
    assert "LEFT VIEW" in pdf_text
    assert "TOP VIEW" in pdf_text
    assert "BOTTOM VIEW" in pdf_text
    assert "Tupperware Products S.A." in pdf_text
