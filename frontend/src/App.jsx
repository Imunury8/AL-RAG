import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Send, FileText, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [isAnswering, setIsAnswering] = useState(false);
  const abortControllerRef = useRef(null);

  // 1. 파일 업로드 핸들러
  const handleUpload = async (selectedFile) => {
    const fileToUpload = selectedFile || file;
    if (!fileToUpload) return alert("ZIP 파일을 선택해주세요.");

    const formData = new FormData();
    formData.append("file", fileToUpload);

    setIsUploading(true);
    abortControllerRef.current = new AbortController();

    try {
      const res = await axios.post(`${API_BASE}/upload-zip/`, formData, {
        signal: abortControllerRef.current.signal
      });
      alert(res.data.message);
    } catch (err) {
      if (axios.isCancel(err)) {
        alert("분석이 중지되었습니다.");
      } else {
        alert("업로드 실패: " + (err.response?.data?.detail || err.message));
      }
    } finally {
      setIsUploading(false);
      abortControllerRef.current = null;
    }
  };

  const handleStopUpload = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  // 2. 질문 전송 핸들러
  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    const userMsg = { role: 'user', content: question };
    setChatHistory(prev => [...prev, userMsg]);
    setIsAnswering(true);
    setQuestion("");

    try {
      const res = await axios.post(`${API_BASE}/ask/`, { question });
      const aiMsg = {
        role: 'ai',
        content: res.data.answer,
        citations: res.data.citations
      };
      setChatHistory(prev => [...prev, aiMsg]);
    } catch (err) {
      setChatHistory(prev => [...prev, { role: 'ai', content: "에러가 발생했습니다." }]);
    } finally {
      setIsAnswering(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8 font-sans">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* <h1 className="text-3xl font-bold text-blue-600">ALZip + AI 문서 에이전트</h1> */}

        {/* 업로드 섹션 */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
          <div className="flex items-center gap-4">
            <input
              type="file"
              accept=".zip"
              onChange={(e) => {
                const selected = e.target.files[0];
                if (selected) {
                  setFile(selected);
                  handleUpload(selected);
                }
              }}
              className="flex-1 text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-s file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            {isUploading ? (
              <button
                onClick={handleStopUpload}
                className="flex items-center gap-2 bg-red-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-red-700"
              >
                <Loader2 className="animate-spin" size={18} />
                중지하기
              </button>
            ) : (
              <button
                onClick={() => handleUpload(file)}
                disabled={!file}
                className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 disabled:bg-blue-300"
              >
                <Upload size={18} />
                분석하기
              </button>
            )}
          </div>
        </div>

        {/* 채팅 섹션 */}
        <div className="bg-white h-[500px] rounded-xl shadow-sm border border-gray-200 flex flex-col">
          <div className="flex-1 overflow-y-auto p-6 space-y-4 no-scrollbar">
            {chatHistory.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] text-xs p-3 rounded-2xl ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800'}`}>
                  <div className="whitespace-pre-wrap text-left prose prose-sm max-w-none">
                    <ReactMarkdown>{msg.content || ""}</ReactMarkdown>
                  </div>
                  {msg.citations && (
                    <div className="mt-3 pt-2 border-t border-gray-200 text-xs text-gray-500">
                      <p className="font-bold mb-1">참조된 출처:</p>
                      {msg.citations.map((c, i) => (
                        <span key={i} className="inline-block bg-white px-2 py-1 rounded mr-1 mb-1 border border-gray-300">
                          {c.file_name} (p.{c.page})
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isAnswering && <div className="text-gray-400 animate-pulse">AI가 문서를 읽고 답변을 생성 중입니다...</div>}
          </div>

          <form onSubmit={handleAsk} className="p-4 border-t flex gap-2">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="문서에 대해 무엇이든 물어보세요..."
              className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button type="submit" className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
              <Send size={20} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;