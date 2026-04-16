import os
import shutil
import zipfile
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv

# --- 추가: .env 파일 로드 ---
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- 추가: OpenAI 임베딩 및 Chroma DB 모듈 ---
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

app = FastAPI()

# 전역 변수로 벡터 DB 인스턴스를 보관 (단일 세션 MVP용)
vector_store = None

def process_and_chunk_files(file_paths):
    documents = []
    for path in file_paths:
        try:
            if path.lower().endswith('.pdf'):
                loader = PyPDFLoader(path)
                documents.extend(loader.load())
            elif path.lower().endswith('.txt'):
                loader = TextLoader(path, encoding='utf-8')
                documents.extend(loader.load())
        except Exception as e:
            print(f"파일 읽기 실패 ({path}): {e}")
            continue

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

@app.post("/upload-zip/")
async def upload_and_extract_zip(file: UploadFile = File(...)):
    global vector_store
    
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="ZIP 파일만 업로드 가능합니다.")

    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, file.filename)

    try:
        # 1. 압축 파일 저장 및 해제
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 2. 타겟 파일 필터링
        target_files = []
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith(('.pdf', '.txt')):
                    target_files.append(os.path.join(root, f))

        if not target_files:
             raise HTTPException(status_code=400, detail="압축 파일 내에 PDF나 TXT 파일이 없습니다.")

        # 3. 텍스트 추출 및 청킹
        chunks = process_and_chunk_files(target_files)

        # --- 4. 추가된 로직: OpenAI 임베딩 후 ChromaDB에 적재 ---
        # 가성비와 속도가 좋은 text-embedding-3-small 모델 사용
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 프로젝트 폴더 내부에 'chroma_db'라는 폴더를 만들어 데이터베이스 저장
        persist_directory = "./chroma_db"
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # 데이터 처리가 모두 끝났으므로 보안과 용량을 위해 임시 폴더 삭제
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "message": "임베딩 및 Vector DB 적재 완료!",
            "total_chunks_embedded": len(chunks),
            "db_directory": persist_directory
        }

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류 발생: {str(e)}")