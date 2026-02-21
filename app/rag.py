# app/rag.py
import os
import time
import random
import requests
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

knowledge_progress = {}

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_URL = "https://api.siliconflow.cn/v1/rerank"


def get_embeddings():
    """初始化 Embedding 模型，使用硅基流动 BAAI/bge-m3"""
    return OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url="https://api.siliconflow.cn/v1",
        chunk_size=50,
    )


def get_vector_store():
    """初始化并获取本地 ChromaDB 向量库实例"""
    persist_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
    return Chroma(
        collection_name="bytecreator_knowledge",
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )


def add_to_knowledge_base(text: str, source: str = "manual_input"):
    """带自动重试机制的入库引擎"""
    print(f"📚 正在处理并切分文本，来源: {source}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    valid_chunks = [chunk for chunk in text_splitter.split_text(text) if chunk.strip()]
    documents = [Document(page_content=chunk, metadata={"source": source}) for chunk in valid_chunks]

    vector_store = get_vector_store()
    batch_size = 50
    total_docs = len(documents)
    knowledge_progress[source] = {"current": 0, "total": total_docs, "status": "processing"}
    print(f"📦 共计 {total_docs} 个有效知识块，开始分批安全入库...")

    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        max_retries = 5
        for attempt in range(max_retries):
            try:
                vector_store.add_documents(batch)
                knowledge_progress[source]["current"] = min(i + batch_size, total_docs)
                print(f"✅ 入库进度: {min(i + batch_size, total_docs)} / {total_docs}")
                time.sleep(0.5)
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (2**attempt) + random.random()
                    print(f"⚠️ 触发限流，等待 {wait_time:.1f}s 后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ 批量入库失败: {e}")
                    raise

    knowledge_progress[source]["status"] = "completed"
    print(f"✅ 成功将 {total_docs} 个文本块存入 ChromaDB 向量库！(保存在 chroma_db 目录)")
    return total_docs


def _rerank_documents(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """调用硅基流动 BAAI/bge-reranker-v2-m3 对候选文档打分并返回 top_k 条"""
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        return docs[:top_k]
    documents_text = [d.page_content for d in docs]
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents_text,
        "top_n": top_k,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(RERANK_URL, json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            return docs[:top_k]
        data = resp.json()
        results = data.get("results") or []
        if not results:
            return docs[:top_k]
        # results 按相关性从高到低，每项含 index（在 docs 中的下标）
        indices = [r["index"] for r in results[:top_k] if 0 <= r["index"] < len(docs)]
        return [docs[i] for i in indices]
    except Exception as e:
        print(f"⚠️ Rerank API 调用失败，回退为向量前 k 条: {e}")
        return docs[:top_k]


def query_knowledge_base(query: str, k: int = 3) -> str:
    """海选 + 精选 (Rerank) 检索架构，重排序失败时回退为前 k 条"""
    try:
        vector_store = get_vector_store()
        initial_results = vector_store.similarity_search(query, k=10)
        if not initial_results:
            print("⚠️ 知识库中未找到高度相关的片段。")
            return ""

        print(f"🔍 [Rerank] 正在对 {len(initial_results)} 条候选知识进行精选...")
        try:
            final_docs = _rerank_documents(query, initial_results, k)
        except Exception as rerank_err:
            print(f"⚠️ 重排序失败，回退为原始前 {k} 条: {rerank_err}")
            final_docs = initial_results[:k]

        context_pieces = [
            f"【来源: {d.metadata.get('source', '未知')}】\n{d.page_content}" for d in final_docs
        ]
        print(f"✅ 检索完成，返回 {len(final_docs)} 个相关片段。")
        return "\n\n---\n\n".join(context_pieces)

    except Exception as e:
        print(f"❌ 检索异常: {e}")
        return ""
