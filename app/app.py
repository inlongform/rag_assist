from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from nomic import embed
from ollama import Client
import gradio as gr
import torch
import re

# ------------------------------
# CONFIGURATION
# ------------------------------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "isaac_chunks"
TOP_K = 10  # Increased from 5 to get more context
VECTOR_DIM = 512  # Must match the dimension used when embedding documents
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# Initialize clients
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
ollama_client = Client()

# ------------------------------
# FUNCTIONS
# ------------------------------
def retrieve_chunks(query: str, top_k: int = TOP_K) -> List[Dict]:
    """Retrieve top-k chunks from Qdrant for a given query."""
    # Embed the query - MUST specify dimensionality=512 to match indexed docs
    resp = embed.text(
        texts=[query],
        model="nomic-embed-text-v1.5",
        task_type="search_query",
        dimensionality=VECTOR_DIM,  # Explicitly set to 512 to match collection
        inference_mode="local",
        device=DEVICE
    )
    query_vector = resp["embeddings"][0]

    # Search with lower threshold to get more results
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        score_threshold=0.5  # Much lower threshold (was 8.82 which is too high)
    )

    # Return both text and score for debugging
    return [
        {
            "text": point.payload.get("text", ""),
            "score": point.score,
            "metadata": point.payload
        }
        for point in results.points
    ]


def filter_chunks(chunks: List[Dict]) -> List[Dict]:
    """Filter out legacy omni.isaac imports but preserve Isaac Sim 5.0 content."""
    pattern = re.compile(r"\bomni\.isaac\b")
    filtered = []
    
    for chunk in chunks:
        text = chunk["text"]
        # Only filter if it contains omni.isaac AND doesn't mention it's deprecated
        if pattern.search(text) and "deprecated" not in text.lower():
            continue
        filtered.append(chunk)
    
    return filtered


def rerank_chunks(query: str, chunks: List[Dict]) -> List[Dict]:
    """Simple keyword-based reranking to boost relevant chunks."""
    query_lower = query.lower()
    keywords = set(query_lower.split())
    
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        # Count keyword matches
        keyword_matches = sum(1 for kw in keywords if kw in text_lower)
        # Boost score based on keyword density
        chunk["boosted_score"] = chunk["score"] + (keyword_matches * 0.1)
    
    # Sort by boosted score
    chunks.sort(key=lambda x: x["boosted_score"], reverse=True)
    return chunks


def build_prompt(query: str, chunks: List[Dict]) -> str:
    """Build a structured prompt with context and clear instructions."""
    
    if not chunks:
        return (
            "You are an expert Isaac Sim 5.0 and Isaac Lab coding assistant.\n"
            "No relevant documentation was found.\n"
            "Please respond with: 'I don't have enough context to answer this question accurately.'\n\n"
            f"User question: {query}"
        )

    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks[:7], 1):  # Use top 7 chunks
        score = chunk.get("boosted_score", chunk.get("score", 0))
        context_parts.append(f"[Context {i}] (relevance: {score:.2f})\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are an expert Isaac Sim 5.0 and Isaac Lab coding assistant.

CRITICAL RULES:
1. Use ONLY the provided context below to generate code
2. Modern Isaac Sim 5.0 uses: isaacsim.core, isaaclab.app (NOT omni.isaac.*)
3. If the context doesn't contain enough information, say "I need more context about [specific topic]"
4. Include import statements from the context
5. Provide working, complete code examples
6. Add brief comments explaining key steps

CONTEXT FROM DOCUMENTATION:
{context}

USER QUESTION:
{query}

RESPONSE (provide complete Python code with imports):
"""


def answer_question(query: str, model_choice: str) -> tuple[str, str]:
    """Generate answer and return both code and debug info."""
    
    # Retrieve chunks
    chunks = retrieve_chunks(query, top_k=TOP_K)
    
    # Debug info
    debug_info = f"Retrieved {len(chunks)} chunks\n"
    debug_info += "Top 5 scores: " + ", ".join(f"{c['score']:.3f}" for c in chunks[:5]) + "\n\n"
    
    # Filter legacy content
    filtered_chunks = filter_chunks(chunks)
    debug_info += f"After filtering: {len(filtered_chunks)} chunks\n"
    
    # Rerank
    reranked_chunks = rerank_chunks(query, filtered_chunks)
    
    # Build prompt
    prompt = build_prompt(query, reranked_chunks)
    
    # Generate response
    try:
        response = ollama_client.generate(
            model=model_choice,
            prompt=prompt,
            options={
                "num_predict": 2048,  # Increased for longer examples
                "temperature": 0.2,   # Slightly higher for more natural code
                "top_p": 0.95,
                "repeat_penalty": 1.1
            }
        )
        code = response["response"]
        
        # Clean up code
        code = code.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()
        
        return code, debug_info
        
    except Exception as e:
        return f"Error: {str(e)}", debug_info


# ------------------------------
# GRADIO UI
# ------------------------------
with gr.Blocks(title="Isaac Sim 5.0 RAG Assistant") as iface:
    gr.Markdown("# 🤖 Isaac Sim 5.0 Coding Assistant")
    gr.Markdown("RAG-powered assistant using Qdrant + Nomic embeddings + Ollama")
    
    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., How do I create a wheeled robot with differential drive controller?",
                lines=3
            )
            model_choice = gr.Dropdown(
                choices=["qwen2.5-coder:7b", "qwen3-coder:latest", "codellama:13b"],
                value="qwen3-coder:latest",
                label="Select LLM Model"
            )
            submit_btn = gr.Button("Generate Code", variant="primary")
        
    with gr.Row():
        with gr.Column():
            code_output = gr.Code(
                label="Generated Code",
                language="python",
                lines=20
            )
        with gr.Column():
            debug_output = gr.Textbox(
                label="Debug Info (retrieval stats)",
                lines=20
            )
    
    # Example queries
    gr.Examples(
        examples=[
            ["How do I create a basic simulation with a robot?"],
            ["Show me how to control a differential drive robot"],
            ["How to set up an Isaac Lab environment?"],
            ["Create a wheeled robot with sensors"],
        ],
        inputs=query_input
    )
    
    submit_btn.click(
        fn=answer_question,
        inputs=[query_input, model_choice],
        outputs=[code_output, debug_output]
    )

if __name__ == "__main__":
    iface.launch(share=False)