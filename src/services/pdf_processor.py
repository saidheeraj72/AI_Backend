from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import camelot
import PyPDF2
from langchain.docstore.document import Document


class PDFProcessor:
    """Handle PDF parsing into LangChain documents."""

    def __init__(self, *, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def extract_text(self, pdf_path: Path) -> str:
        """Extract raw text from a PDF file."""
        text = ""
        try:
            with pdf_path.open("rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            self.logger.info("Extracted text from %s", pdf_path.name)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to extract text from %s: %s", pdf_path.name, exc)
        return text

    def extract_tables(self, pdf_path: Path) -> str:
        """Extract tables from a PDF file using Camelot."""
        table_text = ""
        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
            for index, table in enumerate(tables, start=1):
                table_text += f"\n\nTable {index}:\n{table.df.to_string()}"
            self.logger.info("Extracted %s tables from %s", len(tables), pdf_path.name)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Table extraction skipped for %s: %s", pdf_path.name, exc)
        return table_text

    def process_pdf(self, pdf_path: Path) -> Optional[Document]:
        """Combine text and tables into a LangChain document."""
        text_content = self.extract_text(pdf_path)
        table_content = self.extract_tables(pdf_path)
        combined_content = f"{text_content}{table_content}".strip()

        if not combined_content:
            self.logger.warning("PDF %s produced no readable content", pdf_path.name)
            return None

        return Document(page_content=combined_content, metadata={"source": pdf_path.name})
