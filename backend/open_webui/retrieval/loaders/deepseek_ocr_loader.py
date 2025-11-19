"""
DeepSeek OCR Loader for LangChain
PyPDFLoader와 동일한 인터페이스를 제공하는 OCR 로더
PDF와 이미지 파일을 모두 지원합니다.
"""
import requests
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class DeepSeekOCRLoader(BaseLoader):
    """
    DeepSeek OCR 서버를 사용하여 PDF/이미지를 OCR 처리하고 Document를 반환하는 로더
    
    PyPDFLoader와 동일한 방식으로 사용 가능:
        loader = DeepSeekOCRLoader(
            file_path="document.pdf",
            ocr_server_url="http://localhost:30100",
            extract_images=True
        )
        documents = loader.load()
    """
    
    def __init__(
        self,
        file_path: str,
        ocr_server_url: str = "http://localhost:30100",
        extract_images: Optional[bool] = None,
        **kwargs
    ):
        """
        Args:
            file_path: OCR 처리할 PDF 또는 이미지 파일 경로
            ocr_server_url: OCR 서버 URL (기본값: http://localhost:30100)
            extract_images: PyPDFLoader 호환을 위한 옵션
            **kwargs: 추가 옵션
        """
        self.file_path = file_path
        self.ocr_server_url = ocr_server_url.rstrip("/")
        self.extract_images = extract_images
        self.kwargs = kwargs
    
    def _is_pdf(self, file_path: str) -> bool:
        """파일이 PDF인지 확인"""
        return file_path.lower().endswith('.pdf')
    
    def _pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """PDF를 페이지별 이미지로 변환"""
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF (fitz)가 설치되지 않았습니다. pip install PyMuPDF를 실행하세요.")
        
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # DPI를 높게 설정하여 고품질 이미지 생성
            mat = fitz.Matrix(2.0, 2.0)  # 2x 확대 (약 144 DPI)
            pix = page.get_pixmap(matrix=mat)
            
            if HAS_PIL:
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            else:
                # PIL이 없으면 bytes로 반환
                img = pix.tobytes("png")
            
            images.append(img)
        
        doc.close()
        return images
    
    def _ocr_image(self, image: Image.Image, page_num: int = 1) -> Document:
        """단일 이미지를 OCR 처리하여 Document 반환"""
        import io
        
        # PIL Image를 bytes로 변환
        img_bytes = io.BytesIO()
        if isinstance(image, Image.Image):
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            files = {"file": (f"page_{page_num}.png", img_bytes, "image/png")}
        else:
            # bytes인 경우
            files = {"file": (f"page_{page_num}.png", image, "image/png")}
        
        response = requests.post(
            f"{self.ocr_server_url}/ocr",
            files=files,
            timeout=300
        )
        response.raise_for_status()
        result = response.json()
        
        # OCR 결과에서 텍스트 추출
        documents = result.get("documents", [])
        if documents:
            doc_data = documents[0]
            page_content = doc_data.get("page_content", "")
            metadata = doc_data.get("metadata", {})
        else:
            # documents 형식이 아닌 경우 직접 텍스트 추출 시도
            page_content = result.get("text", result.get("content", ""))
            metadata = result.get("metadata", {})
        
        # PyPDFLoader 형식의 메타데이터 추가
        metadata.update({
            "source": self.file_path,
            "page": page_num,
            "extract_images": self.extract_images
        })
        
        return Document(
            page_content=page_content,
            metadata=metadata
        )
    
    def load(self) -> List[Document]:
        """
        PDF 또는 이미지를 OCR 처리하여 Document 리스트를 반환
        PyPDFLoader와 동일한 형식으로 반환 (페이지별 Document)
        
        Returns:
            List[Document]: OCR 처리된 텍스트를 포함한 Document 리스트
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        documents = []
        
        try:
            if self._is_pdf(self.file_path):
                # PDF 처리: 각 페이지를 이미지로 변환 후 OCR
                images = self._pdf_to_images(self.file_path)
                
                for page_num, image in enumerate(images, start=1):
                    try:
                        doc = self._ocr_image(image, page_num)
                        documents.append(doc)
                    except Exception as e:
                        # 페이지 처리 실패 시 빈 Document 추가
                        documents.append(Document(
                            page_content="",
                            metadata={
                                "source": self.file_path,
                                "page": page_num,
                                "error": str(e)
                            }
                        ))
            else:
                # 이미지 파일 처리
                if HAS_PIL:
                    image = Image.open(self.file_path).convert("RGB")
                    doc = self._ocr_image(image, page_num=1)
                    documents.append(doc)
                else:
                    # PIL이 없으면 직접 파일 전송
                    with open(self.file_path, "rb") as f:
                        files = {"file": (os.path.basename(self.file_path), f, "image/png")}
                        response = requests.post(
                            f"{self.ocr_server_url}/ocr",
                            files=files,
                            timeout=300
                        )
                        response.raise_for_status()
                        result = response.json()
                    
                    # Document 리스트로 변환
                    for doc_data in result.get("documents", []):
                        document = Document(
                            page_content=doc_data.get("page_content", ""),
                            metadata={
                                **doc_data.get("metadata", {}),
                                "source": self.file_path,
                                "page": 1,
                                "extract_images": self.extract_images
                            }
                        )
                        documents.append(document)
            
            return documents
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"OCR server request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to load document: {str(e)}")
