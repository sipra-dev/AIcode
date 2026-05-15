# legalAI_backend.py

import os
import json

from dotenv import load_dotenv

from openai import OpenAI

from typing import TypedDict

from langgraph.graph import StateGraph

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from sentence_transformers import SentenceTransformer, util


# ==========================================
# LOAD ENV
# ==========================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# ==========================================
# LLM FUNCTIONS
# ==========================================

def call_llm(
    system_prompt,
    user_prompt,
    model="gpt-4o",
    max_tokens=1000
):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content


def gpt4o(system_prompt, user_prompt):

    return call_llm(
        system_prompt,
        user_prompt,
        model="gpt-4o",
        max_tokens=2000
    )


def gpt41(system_prompt, user_prompt):

    return call_llm(
        system_prompt,
        user_prompt,
        model="gpt-4.1",
        max_tokens=2000
    )


# ==========================================
# EMBEDDINGS
# ==========================================

class OpenAIEmbeddings(Embeddings):

    def embed_query(self, text):

        return client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

    def embed_documents(self, texts):

        embeddings = []

        for text in texts:

            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            ).data[0].embedding

            embeddings.append(emb)

        return embeddings


# ==========================================
# RERANKER MODEL
# ==========================================

reranker_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)


# ==========================================
# MAIN PIPELINE
# ==========================================

def build_pipeline(pdf_path):

    # ======================================
    # LOAD PDF
    # ======================================

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    # ======================================
    # SPLIT TEXT
    # ======================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(docs)

    # ======================================
    # VECTOR STORE
    # ======================================

    embedding = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embedding=embedding
    )

    # ======================================
    # RERANK FUNCTION
    # ======================================

    def rerank(query, docs, top_k=4):

        texts = [d.page_content for d in docs]

        query_emb = reranker_model.encode(
            query,
            convert_to_tensor=True
        )

        doc_embs = reranker_model.encode(
            texts,
            convert_to_tensor=True
        )

        scores = util.cos_sim(
            query_emb,
            doc_embs
        )[0]

        ranked = sorted(
            zip(texts, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [text for text, _ in ranked[:top_k]]

    # ======================================
    # RETRIEVE FUNCTION
    # ======================================

    def retrieve(query, k=12):

        docs = vectorstore.similarity_search(
            query,
            k=k
        )

        reranked_chunks = rerank(
            query,
            docs,
            top_k=4
        )

        return reranked_chunks

    # ======================================
    # FIELD EXTRACTION
    # ======================================

    def process_field(query):

        retrieved_chunks = retrieve(query)

        system_prompt = """
You are a legal AI assistant.

Use ONLY the provided context.

Do NOT hallucinate.

If information is missing,
reply ONLY with:
Not found
"""

        user_prompt = f"""
Query:
{query}

Context:
{chr(10).join(retrieved_chunks)}
"""

        try:

            response = gpt4o(
                system_prompt,
                user_prompt
            )

            return response.strip()

        except Exception as e:

            return f"Error: {str(e)}"

    # ======================================
    # SUMMARY
    # ======================================

    def process_summary():

        retrieved_chunks = retrieve(
            "Summarize the case"
        )

        system_prompt = """
You are a legal summarization expert.

Use ONLY the provided context.

Generate a concise legal summary of the document provided.
"""

        user_prompt = f"""
Context:
{chr(10).join(retrieved_chunks)}
"""

        try:

            response = gpt4o(
                system_prompt,
                user_prompt
            )

            return response.strip()

        except Exception as e:

            return f"Error: {str(e)}"

    # ======================================
    # JUDGEMENT
    # ======================================

    def process_judgement():

        retrieved_chunks = retrieve(
            "What is the judgement?"
        )

        system_prompt = """
You are a senior legal AI assistant.

Use ONLY the provided context.

Provide:
1. Legal reasoning
2. IPC sections
3. Final verdict
"""

        user_prompt = f"""
Context:
{chr(10).join(retrieved_chunks)}
"""

        try:

            response = gpt41(
                system_prompt,
                user_prompt
            )

            return response.strip()

        except Exception as e:

            return f"Error: {str(e)}"

    # ======================================
    # LANGGRAPH STATE
    # ======================================

    class State(TypedDict):

        ipc: str
        incident: str
        victim_age: str
        place: str
        summary: str
        judgement: str

    # ======================================
    # GRAPH
    # ======================================

    graph = StateGraph(State)

    graph.add_node(
        "ipc",
        lambda s: {
            "ipc":
                process_field(
                    "Extract IPC sections"
                )
        }
    )

    graph.add_node(
        "incident",
        lambda s: {
            "incident":
                process_field(
                    "Describe the incident"
                )
        }
    )

    graph.add_node(
        "victim_age",
        lambda s: {
            "victim_age":
                process_field(
                    "Victim age"
                )
        }
    )

    graph.add_node(
        "place",
        lambda s: {
            "place":
                process_field(
                    "Place of occurrence"
                )
        }
    )

    graph.add_node(
        "summary",
        lambda s: {
            "summary":
                process_summary()
        }
    )

    graph.add_node(
        "judgement",
        lambda s: {
            "judgement":
                process_judgement()
        }
    )

    # ======================================
    # GRAPH FLOW
    # ======================================

    graph.set_entry_point("ipc")

    graph.add_edge("ipc", "incident")
    graph.add_edge("ipc", "victim_age")
    graph.add_edge("ipc", "place")
    graph.add_edge("ipc", "summary")
    graph.add_edge("ipc", "judgement")

    graph.set_finish_point("judgement")

    # ======================================
    # COMPILE GRAPH
    # ======================================

    app = graph.compile()

    result = app.invoke({})

    # ======================================
    # FINAL OUTPUT
    # ======================================

    final_output = {
        "ipc_sections":
            result["ipc"],

        "incident":
            result["incident"],

        "victim_age":
            result["victim_age"],

        "place_of_occurrence":
            result["place"],

        "summary":
            result["summary"],

        "judgement":
            result["judgement"]
    }

    return final_output