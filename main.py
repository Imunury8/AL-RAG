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
from langchain_classic.retrievers import EnsembleRetriever
import pickle # BM25 인덱스를 파일로 저장하기 위함

import fitz  # PyMuPDF
import base64
from io import BytesIO
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

import concurrent.futures
import time  # 추가

# 마크다운(Markdown) 파서 도입
import pymupdf4llm

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

import pymupdf4llm # 맨 위에 추가

def process_and_chunk_files(file_paths):
    docs = []
    images_to_summarize = []
    
    for path in file_paths:
        file_name = os.path.basename(path)
        
        if path.lower().endswith(".pdf"):
            # 🔥 1. 표와 텍스트를 마크다운 형식으로 완벽하게 구조화해서 추출
            md_text = pymupdf4llm.to_markdown(path)
            
            # 마크다운 텍스트를 통째로 Document로 추가 (나중에 스플리터가 자름)
            if md_text.strip():
                docs.append(Document(
                    page_content=md_text, 
                    metadata={"source": file_name, "type": "markdown_text"}
                ))
            
            # 🔥 2. 이미지 추출 로직은 기존과 동일하게 유지 (그래프, 차트용)
            pdf_document = fitz.open(path)
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    image_bytes = base_image["image"]
                    
                    if width < 150 or height < 150 or len(image_bytes) < 10240:
                        continue 
                    if width > 1200 or height > 1200:
                        continue
                        
                    images_to_summarize.append((image_bytes, file_name, page_num + 1))
            pdf_document.close()
            
        elif path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"source": file_name}))

    # 멀티스레딩 이미지 요약 로직 (기존과 완전히 동일하게 유지)
    if images_to_summarize:
        print(f"\n🚀 총 {len(images_to_summarize)}개의 유의미한 이미지를 동시에 분석합니다...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_image = {
                executor.submit(summarize_image_with_vision, img[0], img[1], img[2]): img 
                for img in images_to_summarize
            }
            for future in concurrent.futures.as_completed(future_to_image):
                try:
                    result_doc = future.result()
                    if result_doc:
                        docs.append(result_doc)
                except Exception as exc:
                    print(f"에러: {exc}")

    # 청킹 (마크다운 구조가 깨지지 않도록 사이즈를 넉넉하게 1000 이상으로 유지)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks

def summarize_image_with_vision(image_bytes, file_name, page_num):
    """추출된 이미지를 GPT-4o Vision을 이용해 텍스트로 요약합니다."""
    # 이미지를 Base64로 인코딩
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    # Vision 기능을 위해 gpt-4o 모델 사용 (mini보다 시각 능력이 압도적임)
    chat = ChatOpenAI(model="gpt-4o", max_tokens=500, temperature=0)
    
    prompt_text = (
        "이 이미지(또는 표)의 내용을 아주 상세하게 텍스트로 요약해 줘. "
        "검색 엔진이 나중에 이 내용을 찾을 수 있도록, 포함된 숫자, 고유명사, 핵심 데이터를 빠짐없이 나열해야 해."
    )
    
    try:
        time.sleep(1) # 🔥 과부하 방지턱: API 호출 전 1초 대기

        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ]
                )
            ]
        )
        summary_text = f"[이미지/표 요약 데이터]\n{msg.content}"
        
        # 요약된 텍스트를 LangChain Document 객체로 변환하여 반환
        return Document(
            page_content=summary_text,
            metadata={"source": file_name, "page": page_num, "type": "image_summary"}
        )
    except Exception as e:
        print(f"이미지 요약 중 오류 발생: {e}")
        return None

def process_and_chunk_files(file_paths):
    docs = []
    images_to_summarize = [] # 🔥 요약할 이미지들을 모아둘 대기열
    
    for path in file_paths:
        file_name = os.path.basename(path)
        
        if path.lower().endswith(".pdf"):
            pdf_document = fitz.open(path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # A. 텍스트 추출 (기존 유지)
                text = page.get_text()
                if text.strip():
                    docs.append(Document(
                        page_content=text, 
                        metadata={"source": file_name, "page": page_num + 1, "type": "text"}
                    ))
                
                # B. 이미지 추출 및 대기열에 담기 (기다리지 않음!)
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    image_bytes = base_image["image"]
                    
                    # 🔥 1. 극강의 필터링: 작은 로고 거르기 + '거대한 PPT 배경화면' 거르기
                    # (보통 가로나 세로가 1000px을 넘어가면 슬라이드 배경 템플릿입니다)
                    if width < 150 or height < 150 or len(image_bytes) < 10240:
                        continue 
                    if width > 1200 or height > 1200:
                        continue
                    
                    images_to_summarize.append((image_bytes, file_name, page_num + 1))
            
            pdf_document.close()
            
        elif path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append(Document(
                    page_content=text, 
                    metadata={"source": file_name, "page": 1, "type": "text"}
                ))

    # 🔥 C. 대기열에 모인 이미지를 '멀티스레딩'으로 한 번에(동시에) 요약
    if images_to_summarize:
        print(f"\n🚀 총 {len(images_to_summarize)}개의 유의미한 이미지를 동시에 분석합니다...")
        
        # max_workers=10 이면 10개의 이미지를 동시에 처리합니다.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 모든 작업을 스레드 풀에 던짐
            future_to_image = {
                executor.submit(summarize_image_with_vision, img[0], img[1], img[2]): img 
                for img in images_to_summarize
            }
            
            # 완료되는 대로 받아서 docs에 추가
            for future in concurrent.futures.as_completed(future_to_image):
                try:
                    result_doc = future.result()
                    if result_doc:
                        docs.append(result_doc)
                except Exception as exc:
                    print(f"이미지 병렬 처리 중 에러 발생: {exc}")

    # 문서 청킹 (기존 유지)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
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

        # --- 🔥 5. 신규 추가: BM25 키워드 검색기 생성 및 저장 ---
        bm25_retriever = BM25Retriever.from_documents(chunks)
        # 키워드 검색을 위해 문서 하나당 k값을 3개로 설정
        bm25_retriever.k = 3

        # BM25 데이터는 DB가 아니라 메모리 기반이므로 임시 파일로 저장해둡니다.
        with open("bm25_index.pkl", "wb") as f:
            pickle.dump(bm25_retriever, f)
        
        # 데이터 처리가 모두 끝났으므로 보안과 용량을 위해 임시 폴더 삭제
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "message": "하이브리드 검색 인덱싱(Vector + BM25) 완료!",
            "total_chunks_embedded": len(chunks)
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
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        persist_directory = "./chroma_db"
        
        if not os.path.exists(persist_directory):
            raise HTTPException(status_code=400, detail="DB가 없습니다. 파일을 먼저 업로드하세요.")
            
        # 3. embedding_function 파라미터 확인
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        vector_retriever = db.as_retriever(search_kwargs={"k": 10})

        if not os.path.exists("bm25_index.pkl"):
            raise HTTPException(status_code=400, detail="BM25 인덱스가 없습니다.")
            
        with open("bm25_index.pkl", "rb") as f:
            bm25_retriever = pickle.load(f)
        
        bm25_retriever.k = 10

        # 하이브리드 검색기 가동
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever], 
            weights=[0.5, 0.5]
        )

        # 2. 질문과 가장 관련성이 높은 문서 조각(Chunk) 3개 검색 (Retriever)
        docs = ensemble_retriever.invoke(req.question)

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
            "당신은 제공된 기업 실적 문서와 마크다운(Markdown) 표를 완벽하게 해독하는 수석 데이터 분석가입니다.\n"
            "아래 제공된 [Context]를 바탕으로 질문에 답변하세요.\n\n"
            "🚨 [분석 지침]\n"
            "1. 표(Table) 데이터가 있다면 행(Row)과 열(Column)의 맥락을 꼼꼼히 파악하여 숫자를 읽어내세요.\n"
            "2. 여러 분기(1Q, 2Q 등)의 데이터가 흩어져 있다면, 이를 종합하거나 합산하여 사용자가 원하는 단위(예: 연간 실적)로 정리해서 답변하세요.\n"
            "3. '매출', '영업이익', '순이익' 등의 핵심 재무 지표를 헷갈리지 마세요.\n"
            "4. 답변 내용의 끝에는 반드시 참조한 문서의 [출처(파일명)]를 명시하세요.\n"
            "5. 제공된 [Context] 안의 내용으로 도저히 유추할 수 없는 질문에만 '문서에서 내용을 찾을 수 없습니다.'라고 답변하세요.\n\n"
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