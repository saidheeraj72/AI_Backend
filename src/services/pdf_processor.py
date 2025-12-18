from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import camelot
import PyPDF2
from langchain_core.documents import Document


class PDFProcessor:
    """Handle PDF parsing into LangChain documents."""

    def __init__(self, *, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def get_page_count(self, pdf_path: Path) -> int:
        """Get the total number of pages in the PDF."""
        try:
            with pdf_path.open("rb") as file:
                reader = PyPDF2.PdfReader(file)
                return len(reader.pages)
        except Exception as exc:
            self.logger.error("Failed to get page count for %s: %s", pdf_path.name, exc)
            return 0

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

    def extract_tables_batch(self, pdf_path: Path, pages: str) -> str:
        """Extract tables from a specific page range using Camelot."""
        table_text = ""
        try:
            # camelot.read_pdf takes a file path string
            tables = camelot.read_pdf(str(pdf_path), pages=pages, flavor="lattice")
            for i, table in enumerate(tables):
                # We don't have a global counter here, so we just append tables.
                # The caller might want to handle numbering, or we just dump the content.
                table_text += f"\n\nTable (Pages {pages}):\n{table.df.to_string()}"
        except Exception as batch_exc:
            self.logger.warning("Table extraction failed for pages %s in %s: %s", pages, pdf_path.name, batch_exc)
        return table_text

    def extract_tables(self, pdf_path: Path) -> str:
        """Extract tables from a PDF file using Camelot in batches."""
        table_text = ""
        try:
            num_pages = self.get_page_count(pdf_path)
            if num_pages == 0:
                return ""

            # Process in chunks of 10 pages to save memory
            batch_size = 10
            total_tables = 0

            for start in range(1, num_pages + 1, batch_size):
                end = min(start + batch_size - 1, num_pages)
                pages_range = f"{start}-{end}"
                
                chunk_text = self.extract_tables_batch(pdf_path, pages_range)
                table_text += chunk_text
                # Note: The original numbering "Table {total_tables}" logic is slightly lost 
                # if we rely purely on batch method without state, but for LLM context 
                # the exact sequential number matters less than the content. 
                # If strictly needed, we could parse the output or handle it in the loop.

            self.logger.info("Extracted tables from %s (approx)", pdf_path.name)
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
