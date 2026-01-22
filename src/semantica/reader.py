import os
import pypdf
import docx
from .config import FILE_TYPES


class FileReader:
    def __init__(self):
        self.text_exts = FILE_TYPES['text']
        self.code_exts = FILE_TYPES['code']
        self.supported_exts = self.text_exts | self.code_exts


    def is_supported(self, filepath):
        return os.path.splitext(filepath)[1].lower() in self.supported_exts


    def get_file_type(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        if ext in self.code_exts:
            return 'code'
        if ext in self.text_exts:
            return 'text'
        return None


    def read(self, filepath):
        if not self.is_supported(filepath):
            return None

        ext = os.path.splitext(filepath)[1].lower()
        ftype = self.get_file_type(filepath)
        
        try:
            if ext == '.pdf':
                content = self._read_pdf(filepath)
            elif ext in ['.docx', '.doc']:
                content = self._read_docx(filepath)
            else:
                content = self._read_plain_text(filepath)
                
            if not content.strip():
                return None

            return {
                "type": ftype,
                "content": content
            }
        except Exception as e:
            return {"error": str(e)}


    def _read_plain_text(self, filepath):
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()


    def _read_pdf(self, filepath):
        text = []
        try:
            reader = pypdf.PdfReader(filepath)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text.append(extracted)
        except:
            return ""
        return "\n".join(text)


    def _read_docx(self, filepath):
        text = []
        try:
            doc = docx.Document(filepath)
            for para in doc.paragraphs:
                if para.text:
                    text.append(para.text)
        except:
            return ""
        return "\n".join(text)