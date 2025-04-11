import os
import PyPDF2

class DocumentLoader:
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir

    def load_pdfs(self):
        """Charge le texte des PDFs dans le dossier raw."""
        documents = {}
        for file in os.listdir(self.raw_dir):
            if file.endswith(".pdf"):
                path = os.path.join(self.raw_dir, file)
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    documents[file] = text
        return documents
