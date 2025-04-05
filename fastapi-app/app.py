from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid
from cvpipeline import process_pdf

app = FastAPI()

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Save the uploaded PDF to a temporary location.
    temp_filename = f"temp_{uuid.uuid4().hex}.pdf"
    temp_file_path = os.path.join("/tmp", temp_filename)
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process the PDF using the embedding pipeline.
        process_pdf(temp_file_path)
        return {"status": "success", "message": "PDF processed and embeddings created."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        os.remove(temp_file_path)
