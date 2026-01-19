import os
import pypdf
import docx

CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', 
    '.go', '.rs', '.php', '.rb', '.sh', '.sql', '.html', '.css', 
    '.json', '.yaml', '.yml', '.xml', '.dockerfile', '.md', '.txt'
}


class FileReader:
    def read(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in CODE_EXTENSIONS:
            return {
                "type": "code",
                "content": self._read_text(filepath)
            }
            
        elif ext == '.pdf':
            return {
                "type": "text",
                "content": self._read_pdf(filepath)
            }
            
        elif ext == '.docx':
            return {
                "type": "text",
                "content": self._read_docx(filepath)
            }
            
        return None

    def _read_text(self, filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _read_pdf(self, filepath):
        text = []
        reader = pypdf.PdfReader(filepath)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text.append(content)
        return "\n".join(text)

    def _read_docx(self, filepath):
        doc = docx.Document(filepath)
        text = []
        for para in doc.paragraphs:
            if para.text:
                text.append(para.text)
        return "\n".join(text)