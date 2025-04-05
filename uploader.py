import requests
import sys

def upload_pdf(file_path, url="http://localhost:8000/upload-pdf"):
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
            response.raise_for_status()
            print("Upload successful!")
            print("Response:", response.json())
    except Exception as e:
        print(f"Error uploading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python uploader.py path/to/file.pdf")
    else:
        file_path = sys.argv[1]
        upload_pdf(file_path)
