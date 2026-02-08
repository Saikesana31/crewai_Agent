# Here we will create a custom tool for the crewai framework

from crewai.tools import BaseTool
from pypdf import PdfReader
from typing import List, Dict, Any

class PDFSearchTool(BaseTool):
    """A tool for searching PDFs."""
    name: str = "PDF Reader"
    description: str = "A tool for searching PDFs."
    
    def _run(self, file_path: str) -> str:
    
        """ Read the PDF file and return the text."""
        try:
            reader = PdfReader(file_path)
            texts = []
            print(f"Reading PDF file: {file_path}")
            for page in reader.pages:
                texts.append(page.extract_text())
            return "\n\n".join(texts)
        except Exception as e:
            return f"Error reading PDF file: {e}"


