import os
import shutil
import zipfile
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from fastapi.middleware.cors import CORSMiddleware

# --- 추가: .env 파일 로드 ---
load_dotenv(override=True)

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- 추가: OpenAI 임베딩 및 Chroma DB 모듈 ---
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pickle # BM25 인덱스를 파일로 저장하기 위함

app = FastAPI()

# FastAPI 객체 생성 직후에 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 개발 환경에서는 모두 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# 사용자가 보낼 질문 형식을 정의하는 모델
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(req: QuestionRequest):
    try:
        # 1. 저장된 Vector DB 불러오기
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        persist_directory = "./chroma_db"
        
        # DB 폴더가 있는지 먼저 확인
        if not os.path.exists(persist_directory):
            raise HTTPException(status_code=400, detail="Vector DB가 없습니다. 먼저 ZIP 파일을 업로드해주세요.")
            
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        # 2. 질문과 가장 관련성이 높은 문서 조각(Chunk) 3개 검색 (Retriever)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(req.question)

        if not docs:
            return {"answer": "관련된 문서를 찾을 수 없습니다.", "citations": []}

        # 3. 검색된 문서 내용과 출처(메타데이터)를 LLM에게 줄 컨텍스트로 정리
        context_text = ""
        citations = []
        
        for i, doc in enumerate(docs):
            # 경로에서 파일명만 깔끔하게 추출
            source_path = doc.metadata.get("source", "알 수 없는 문서")
            file_name = os.path.basename(source_path)
            page = doc.metadata.get("page", "알 수 없음")
            
            context_text += f"\n[문서 {i+1}] 출처: {file_name} (페이지/위치: {page})\n내용: {doc.page_content}\n"
            
            # 프론트엔드에서 활용하기 좋게 출처만 따로 배열로 저장
            citations.append({"file_name": file_name, "page": page})

        # 4. LLM 세팅 및 프롬프트 작성 (환각 방지 강조)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # temperature=0으로 설정해 창의성보다 정확도 우선
        
        prompt_template = ChatPromptTemplate.from_template(
            "당신은 제공된 문서만을 기반으로 답변하는 스마트 AI 어시스턴트입니다.\n"
            "아래 제공된 [Context]를 바탕으로 질문에 답변하세요.\n"
            "답변 내용의 끝에는 반드시 참조한 문서의 [출처(파일명 및 페이지)]를 요약해서 명시하세요.\n"
            "만약 [Context]에 질문에 대한 명확한 답이 없다면, 지어내지 말고 '제공된 문서에서 내용을 찾을 수 없습니다.'라고 답변하세요.\n\n"
            "[Context]\n{context}\n\n"
            "질문: {question}"
        )
        
        # 5. 체인(Chain) 실행하여 최종 답변 얻기
        chain = prompt_template | llm
        response = chain.invoke({
            "context": context_text, 
            "question": req.question
        })

        # 6. 최종 결과 반환
        return {
            "question": req.question,
            "answer": response.content,
            "citations": citations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류 발생: {str(e)}")