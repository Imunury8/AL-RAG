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

from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_classic import hub

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

# 🔥 1. RAG 검색 도구 정의
# 🔥 1. RAG 검색 도구 정의 (수정됨)
@tool
def search_company_documents(query: str) -> str:
    """기업의 실적, 매출, 영업이익 등 문서 내의 정보를 검색할 때 사용합니다. 
    질문이 구체적일수록 정확한 정보를 찾아옵니다."""
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = "./chroma_db"
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    vector_retriever = db.as_retriever(search_kwargs={"k": 10})
    with open("bm25_index.pkl", "rb") as f:
        bm25_retriever = pickle.load(f)
        
    bm25_retriever.k = 10  # 🔥 여기서도 k값을 10으로 늘려야 합니다!
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
    )
    
    docs = ensemble_retriever.invoke(query)
    
    # 🔥 에이전트가 출처를 알 수 있도록 텍스트 앞에 파일명을 붙여서 반환
    result_texts = []
    for d in docs:
        source = d.metadata.get("source", "알 수 없는 문서")
        result_texts.append(f"--- [문서 출처: {source}] ---\n{d.page_content}")
        
    return "\n\n".join(result_texts)

# 🔥 2. 계산기 도구 정의
@tool
def calculate(expression: str) -> str:
    """숫자 계산이나 합산, 평균 계산이 필요할 때 사용합니다. 
    파이썬의 eval() 함수처럼 수학 식을 입력받아 결과를 반환합니다."""
    try:
        # 안전한 계산을 위해 파이썬 내장 라이브러리 활용
        return str(eval(expression.replace(',', '')))
    except Exception as e:
        return f"계산 오류: {str(e)}"

# 🔥 3. 에이전트 초기화 (수정됨)
tools = [search_company_documents, calculate]

# OpenAI 에이전트용 표준 프롬프트 불러오기
prompt = hub.pull("hwchase17/openai-tools-agent")

# 🔥 잃어버렸던 수석 분석가 프롬프트를 에이전트의 뇌(System Message)에 이식
prompt.messages[0].prompt.template = (
    "당신은 제공된 기업 실적 문서와 마크다운(Markdown) 표를 완벽하게 해독하는 수석 데이터 분석가입니다.\n"
    "사용자의 질문에 답하기 위해 'search_company_documents' 도구를 사용하여 관련 문서를 꼼꼼히 검색하세요.\n"
    "수학적 계산이나 합산이 필요하다면 당신이 직접 계산하지 말고 반드시 'calculate' 도구를 사용하세요.\n"
    "🚨 [분석 지침]\n"
    "1. 표(Table) 데이터가 있다면 행(Row)과 열(Column)의 맥락을 꼼꼼히 파악하여 숫자를 읽어내세요.\n"
    "2. 검색된 문서 데이터 상단에 있는 [문서 출처: 파일명]을 확인하고, 답변의 끝에 반드시 출처를 명시하세요.\n"
    "3. 제공된 문서에 내용이 없다면 억지로 지어내지 말고 '문서에서 내용을 찾을 수 없습니다'라고 솔직하게 답변하세요."
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def process_and_chunk_files(file_paths):
    docs = []
    images_to_summarize = []
    
    for path in file_paths:
        file_name = os.path.basename(path)
        
        # ==========================================
        # 1. PDF 파일 처리 (마크다운 텍스트 + 이미지 추출)
        # ==========================================
        if path.lower().endswith(".pdf"):
            print(f"📄 PDF 처리 중: {file_name}")
            
            # [A] 텍스트 및 표 처리: pymupdf4llm을 사용해 마크다운으로 깔끔하게 추출
            md_text = pymupdf4llm.to_markdown(path)
            if md_text.strip():
                docs.append(Document(
                    page_content=md_text, 
                    metadata={"source": file_name, "type": "markdown_text"}
                ))
            
            # [B] 이미지/그래프 처리: PyMuPDF(fitz)로 좌표 기반 이미지 탐색
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
                    
                    # 🚨 쓰레기 데이터 필터링 
                    # 너무 작거나(아이콘, 로고), 너무 큰(PPT 배경 템플릿) 이미지는 버림
                    if width < 150 or height < 150 or len(image_bytes) < 10240:
                        continue 
                    if width > 1200 or height > 1200:
                        continue
                        
                    # 당장 API를 쏘지 않고 '대기열 리스트'에 차곡차곡 담아둠
                    images_to_summarize.append((image_bytes, file_name, page_num + 1))
                    
            pdf_document.close()
            
        # ==========================================
        # 2. TXT 파일 처리 (단순 텍스트)
        # ==========================================
        elif path.lower().endswith(".txt"):
            print(f"📝 TXT 처리 중: {file_name}")
            with open(path, "r", encoding="utf-8") as f:
                docs.append(Document(
                    page_content=f.read(), 
                    metadata={"source": file_name, "type": "text"}
                ))

    # ==========================================
    # 3. 이미지 병렬 요약 (멀티스레딩)
    # ==========================================
    if images_to_summarize:
        print(f"\n🚀 총 {len(images_to_summarize)}개의 유의미한 이미지를 병렬 분석합니다...")
        
        # 워커(Worker) 4명을 투입해 대기열의 이미지를 동시에 API로 쏴서 요약
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_image = {
                executor.submit(summarize_image_with_vision, img[0], img[1], img[2]): img 
                for img in images_to_summarize
            }
            
            for future in concurrent.futures.as_completed(future_to_image):
                try:
                    result_doc = future.result() # 요약된 Document 객체 반환
                    if result_doc:
                        docs.append(result_doc)
                except Exception as exc:
                    print(f"❌ 이미지 분석 중 에러 발생: {exc}")

    # ==========================================
    # 4. 문서 청킹 (Chunking)
    # ==========================================
    print(f"\n✂️ 추출된 문서를 청킹합니다...")
    # 마크다운 표 구조가 중간에 잘리지 않도록 chunk_size를 1000 이상으로 넉넉하게 잡음
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    print(f"✅ 총 {len(chunks)}개의 데이터 조각(Chunk)이 완성되었습니다.\n")
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
        # 이제 직접 검색하는 대신 에이전트에게 질문을 던집니다.
        # 에이전트가 질문을 보고 search_company_documents를 쓸지, calculate를 쓸지 결정합니다.
        response = agent_executor.invoke({"input": req.question})
        
        # 에이전트가 최종적으로 내놓은 답변 반환
        return {
            "answer": response["output"],
            "citations": [] # 에이전트 모드에서는 출처 추출 로직을 추가로 커스텀해야 합니다.
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))