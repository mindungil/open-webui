"""
ChandraLoader 클라이언트 버전
기존 ChandraLoader 인터페이스를 유지하며 서버와 통신

기존 코드 변경 없이 사용 가능:
    # 기존 코드
    loader = ChandraLoader(file_path="document.pdf", device="cuda:1")
    
    # 새로운 코드 (동일한 인터페이스)
    loader = ChandraLoader(
        file_path="document.pdf",
        server_url="http://new-server:8000"  # 새로운 서버 URL만 추가
    )
"""

from typing import List, Optional
import os
import requests
from pathlib import Path
from langchain_core.documents import Document
from tqdm import tqdm


class ChandraLoader:
    """
    Chandra OCR 서버 클라이언트
    기존 ChandraLoader와 동일한 인터페이스 제공
    """
    
    def __init__(
        self,
        file_path: str,
        server_url: Optional[str] = None,
        dpi: int = 300,
        timeout: int = 600,
        # 하위 호환성을 위한 파라미터 (무시됨)
        model_path: Optional[str] = None,
        extract_images: bool = True,
        device: str = "cuda:1",
        memory_cleanup_interval: int = 5
    ):
        """
        Args:
            file_path: PDF 파일 경로
            server_url: Chandra OCR 서버 URL (기본값: 환경변수 또는 localhost)
            dpi: PDF to Image 변환 시 해상도
            timeout: HTTP 요청 타임아웃 (초)
            
            # 하위 호환성을 위한 파라미터 (서버 버전에서는 무시됨)
            model_path: 사용되지 않음
            extract_images: 사용되지 않음  
            device: 사용되지 않음
            memory_cleanup_interval: 사용되지 않음
        """
        self.file_path = file_path
        self.dpi = dpi
        self.timeout = timeout
        
        # 서버 URL 설정
        self.server_url = server_url or os.getenv(
            "CHANDRA_SERVER_URL",
            "http://192.168.0.201:30030"
        )
        
        # 파일 존재 확인
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # 서버 연결 확인
        self._check_server_health()
    
    def _check_server_health(self):
        """서버 상태 확인"""
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=10
            )
            response.raise_for_status()
            
            health_data = response.json()
            print(f"Connected to Chandra OCR Server")
            print(f"Status: {health_data['status']}")
            print(f"GPUs: {health_data['gpus_available']}")
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Chandra OCR server at {self.server_url}\n"
                f"Please make sure the server is running."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Chandra OCR server: {str(e)}"
            )
    
    def _send_ocr_request(self) -> dict:
        """서버에 OCR 요청 전송"""
        url = f"{self.server_url}/ocr"
        
        print(f"Uploading PDF to server: {self.file_path}")
        
        with open(self.file_path, "rb") as f:
            files = {
                "file": (Path(self.file_path).name, f, "application/pdf")
            }
            params = {
                "dpi": self.dpi
            }
            
            try:
                response = requests.post(
                    url,
                    files=files,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                return response.json()
                
            except requests.exceptions.Timeout:
                raise TimeoutError(
                    f"OCR request timed out after {self.timeout} seconds.\n"
                    f"Try increasing the timeout parameter."
                )
            except requests.exceptions.HTTPError as e:
                raise RuntimeError(
                    f"OCR request failed: {e.response.status_code}\n"
                    f"Response: {e.response.text}"
                )
            except Exception as e:
                raise RuntimeError(f"OCR request failed: {str(e)}")
    
    def load(self) -> List[Document]:
        """
        PDF를 로드하고 OCR 수행
        LangChain Document 객체 리스트 반환
        
        Returns:
            List[Document]: LangChain Document 객체 리스트
        """
        # 서버에 OCR 요청
        result = self._send_ocr_request()
        
        if not result["success"]:
            raise RuntimeError(
                f"OCR processing failed: {result.get('error', 'Unknown error')}"
            )
        
        # Document 객체로 변환
        documents = []
        total_pages = result["total_pages"]
        
        print(f"Processing {total_pages} pages...")
        
        for page_data in tqdm(result["pages"], desc="Converting to Documents"):
            page_num = page_data["page"]
            
            doc = Document(
                page_content=page_data["content"],
                metadata={
                    "source": self.file_path,
                    "page": page_num,
                    "total_pages": total_pages,
                    "success": page_data["success"],
                    "gpu_id": page_data.get("gpu_id")
                }
            )
            
            if not page_data["success"]:
                doc.metadata["error"] = page_data.get("error")
            
            documents.append(doc)
        
        print(f"Successfully loaded {len(documents)} pages")
        return documents
    
    def lazy_load(self) -> List[Document]:
        """
        Lazy loading (PyPDFLoader 호환)
        실제로는 일반 load와 동일하게 동작
        """
        return self.load()


# 사용 예시
if __name__ == "__main__":
    # 방법 1: 환경변수 사용
    # export CHANDRA_SERVER_URL=http://new-server:8000
    loader = ChandraLoader(
        file_path="path/to/your/document.pdf"
    )
    
    # 방법 2: 명시적 URL 지정
    loader = ChandraLoader(
        file_path="path/to/your/document.pdf",
        server_url="http://192.168.1.100:8000",
        dpi=300,
        timeout=600
    )
    
    # 방법 3: 기존 코드 그대로 사용 (하위 호환)
    loader = ChandraLoader(
        file_path="path/to/your/document.pdf",
        device="cuda:1",  # 무시됨
        memory_cleanup_interval=3  # 무시됨
    )
    
    # 문서 로드
    documents = loader.load()
    
    # 결과 확인
    for doc in documents:
        print(f"\n=== Page {doc.metadata['page'] + 1} ===")
        print(f"GPU: {doc.metadata.get('gpu_id')}")
        print(doc.page_content[:500])
