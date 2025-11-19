"""
DeepSeek-OCR based PDF loader that mimics PyPDFLoader interface.

This loader replaces PyPDFLoader by converting PDF pages to images,
encoding them as base64, and sending them to DeepSeek-OCR API for text extraction.
"""

import base64
import io
import time
from pathlib import Path
from typing import Iterator, Optional, Dict, Any

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI package is required. Install it with: pip install openai"
    )

try:
    import pypdf
    from PIL import Image
except ImportError:
    raise ImportError(
        "pypdf and Pillow are required. Install with: pip install pypdf Pillow"
    )

# Try to import langchain components, but allow usage without it
try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback Document class if langchain is not available
    class Document:
        def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
            self.page_content = page_content
            self.metadata = metadata or {}


class DeepSeekOCRLoader:
    """
    PDF loader using DeepSeek-OCR for text extraction.

    This class provides the same interface as PyPDFLoader but uses
    DeepSeek-OCR API instead of pypdf's text extraction.

    Args:
        file_path: Path to the PDF file (string or Path object)
        api_base_url: DeepSeek-OCR API base URL (default: http://localhost:30040/v1)
        api_key: API key for authentication (default: "EMPTY")
        timeout: Request timeout in seconds (default: 3600)
        ocr_prompt: Prompt to send to OCR model (default: "Free OCR.")
        temperature: Model temperature (default: 0.0)
        max_tokens: Maximum tokens for generation (default: 2048)
        dpi: DPI for PDF to image conversion (default: 200)
        extra_body: Extra parameters for the API request
    """

    def __init__(
        self,
        file_path: str,
        api_base_url: str = "http://localhost:30040/v1",
        api_key: str = "EMPTY",
        timeout: int = 3600,
        ocr_prompt: str = "Free OCR.",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        dpi: int = 200,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize the DeepSeek-OCR loader."""
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.file_path.suffix.lower() == '.pdf':
            raise ValueError(f"File must be a PDF, got: {self.file_path.suffix}")

        # API configuration
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base_url,
            timeout=timeout
        )

        # OCR configuration
        self.ocr_prompt = ocr_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dpi = dpi

        # Default extra_body configuration from DeepSeek-OCR example
        self.extra_body = extra_body or {
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": 30,
                "window_size": 90,
                "whitelist_token_ids": [128821, 128822],  # <td>, </td>
            },
        }

    def _pdf_page_to_base64(self, page: pypdf.PageObject, page_number: int) -> str:
        """
        Convert a PDF page to base64 encoded PNG image.

        Args:
            page: pypdf PageObject
            page_number: Page number for error messages

        Returns:
            base64 encoded PNG image string
        """
        try:
            # Extract images from the page if available
            # This approach tries to render the page as an image
            # Note: pypdf doesn't have built-in page rendering, so we need pdf2image
            # For now, we'll use a workaround with images in the page

            # Better approach: use pdf2image library
            try:
                from pdf2image import convert_from_path

                # Convert single page to image
                images = convert_from_path(
                    self.file_path,
                    dpi=self.dpi,
                    first_page=page_number + 1,  # pdf2image uses 1-indexed pages
                    last_page=page_number + 1
                )

                if not images:
                    raise ValueError(f"Failed to convert page {page_number} to image")

                image = images[0]

            except ImportError:
                # Fallback: try to extract the largest image from the page
                # or create a blank image with extracted text
                raise ImportError(
                    "pdf2image is required for full page rendering. "
                    "Install with: pip install pdf2image\n"
                    "Also requires poppler: apt-get install poppler-utils (Linux) "
                    "or brew install poppler (Mac)"
                )

            # Convert PIL Image to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            return f"data:image/png;base64,{base64_image}"

        except Exception as e:
            raise RuntimeError(f"Failed to convert page {page_number} to image: {str(e)}")

    def _ocr_page(self, base64_image: str, page_number: int) -> str:
        """
        Perform OCR on a base64 encoded image using DeepSeek-OCR API.

        Args:
            base64_image: Base64 encoded image with data URI scheme
            page_number: Page number for logging

        Returns:
            Extracted text from the image
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": self.ocr_prompt
                    }
                ]
            }
        ]

        try:
            start = time.time()
            response = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-OCR",
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body=self.extra_body,
            )
            elapsed = time.time() - start

            text = response.choices[0].message.content
            print(f"Page {page_number}: OCR completed in {elapsed:.2f}s")

            return text

        except Exception as e:
            print(f"Warning: OCR failed for page {page_number}: {str(e)}")
            return f"[OCR Error on page {page_number}: {str(e)}]"

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load and process PDF pages.

        This method mimics PyPDFLoader.lazy_load() interface.

        Yields:
            Document objects with page_content and metadata
        """
        # Open PDF file
        with open(self.file_path, "rb") as pdf_file:
            pdf_reader = pypdf.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)

            print(f"Processing PDF: {self.file_path.name} ({total_pages} pages)")

            # Process each page
            for page_number, page in enumerate(pdf_reader.pages):
                try:
                    # Convert page to base64 image
                    base64_image = self._pdf_page_to_base64(page, page_number)

                    # Perform OCR
                    text = self._ocr_page(base64_image, page_number)

                    # Create Document object (matching PyPDFLoader output)
                    metadata = {
                        "source": str(self.file_path),
                        "page": page_number,
                        "total_pages": total_pages,
                        "ocr_engine": "deepseek-ocr",
                    }

                    yield Document(
                        page_content=text,
                        metadata=metadata
                    )

                except Exception as e:
                    print(f"Error processing page {page_number}: {str(e)}")
                    # Yield error document to maintain page count
                    yield Document(
                        page_content=f"[Error processing page {page_number}: {str(e)}]",
                        metadata={
                            "source": str(self.file_path),
                            "page": page_number,
                            "total_pages": total_pages,
                            "error": str(e),
                        }
                    )

    def load(self) -> list[Document]:
        """
        Load all pages at once.

        Returns:
            List of Document objects
        """
        return list(self.lazy_load())


# Compatibility alias to match PyPDFLoader interface
class PyPDFLoader(DeepSeekOCRLoader):
    """
    Drop-in replacement for langchain's PyPDFLoader using DeepSeek-OCR.

    Usage:
        # Replace this:
        # from langchain_community.document_loaders import PyPDFLoader

        # With this:
        from deepseek_ocr.loader import PyPDFLoader

        loader = PyPDFLoader("document.pdf")
        documents = loader.load()
    """
    pass


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python loader.py <pdf_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Create loader
    loader = DeepSeekOCRLoader(
        file_path=pdf_path,
        api_base_url="http://localhost:30040/v1",
    )

    # Load documents
    print("Loading PDF with DeepSeek-OCR...")
    documents = loader.load()

    # Print results
    print(f"\n{'='*80}")
    print(f"Extracted {len(documents)} pages")
    print(f"{'='*80}\n")

    for doc in documents:
        print(f"Page {doc.metadata['page']}:")
        print(f"{doc.page_content[:500]}...")
        print(f"\n{'-'*80}\n")

