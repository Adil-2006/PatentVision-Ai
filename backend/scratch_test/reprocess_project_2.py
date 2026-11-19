import sys, os
sys.path.insert(0, 'backend')
os.environ['PV_DATABASE_URL'] = 'sqlite:///backend/patentvision.db'
os.environ['PV_STORAGE_ROOT'] = 'backend/storage'
os.environ['U2NET_HOME'] = r'c:\Adil\Patent-Vision\backend\models'

from app.core.database import SessionLocal
from app.models import Project, Image, DocumentMetadata, ProjectDocument
from app.services.processing import _execute, queue_image
from app.services.document_generation import generate_patent_pdf
from app.core.storage import storage
from sqlalchemy import select

db = SessionLocal()

p2_images = db.query(Image).filter(Image.project_id == 2, Image.is_active == True).order_by(Image.id).all()
print(f'Reprocessing {len(p2_images)} images for Project 2...')
for img in p2_images:
    print(f'Queueing & Processing Image {img.id} ({img.view_type})...')
    queue_image(db, img)
    _execute(db, img.id)
    img = db.query(Image).filter(Image.id == img.id).first()
    img.approval_status = 'approved'
    db.commit()
    print(f'   Done Image {img.id}: status={img.processing_status}, quality={img.result.quality_status if img.result else None}')

meta = db.scalar(select(DocumentMetadata).where(DocumentMetadata.project_id == 2))
if not meta:
    meta = DocumentMetadata(project_id=2)
    db.add(meta)

meta.applicant_name = 'RV University'
meta.product_title = 'BRUSHMATE-PRO'
meta.filing_date = '01/07/2025'
meta.novelty_statement = 'Novelty resides in the shape and configuration of the "BRUSHMATE-PRO" as illustrated.'
meta.agent_name = 'Phani Kumar Pullela'
meta.agent_registration_number = 'IN/PA-4726'
meta.agent_designation = 'Agent for the Applicant'

sig_path = 'backend/scratch_test/ref_extracted/image2.jpg'
sig_bytes = None
if os.path.exists(sig_path):
    with open(sig_path, 'rb') as f:
        sig_bytes = f.read()
    sig_key = 'signatures/2/phani_kumar_pullela.jpg'
    storage.save_bytes(sig_key, sig_bytes, 'image/jpeg')
    meta.signature_key = sig_key

db.commit()
db.refresh(meta)

REQUIRED = ['Perspective', 'Front', 'Rear', 'Right', 'Left', 'Top', 'Bottom']
drawings = {}
active_imgs = {img.view_type: img for img in db.query(Image).filter(Image.project_id == 2, Image.is_active == True).all()}
for v in REQUIRED:
    img = active_imgs[v]
    drawings[v] = storage.read_bytes(img.result.patent_drawing_path)

pdf_bytes = generate_patent_pdf(meta, drawings, sig_bytes)
pdf_out_path = 'Project_2_BRUSHMATE_PRO_Patent_7_Views.pdf'
with open(pdf_out_path, 'wb') as f:
    f.write(pdf_bytes)

doc_key = f'documents/2/BRUSHMATE_PRO_Patent_Drawings_v2.pdf'
storage.save_bytes(doc_key, pdf_bytes, 'application/pdf')
doc = ProjectDocument(
    project_id=2,
    version=2,
    storage_key=doc_key,
    filename='BRUSHMATE_PRO_Patent_Drawings_v2.pdf',
    page_count=7,
)
db.add(doc)
db.commit()

print(f'Successfully generated PDF: {pdf_out_path} ({len(pdf_bytes)} bytes)')
db.close()
