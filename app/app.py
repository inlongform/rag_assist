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
        dimensionality=512,  # Explicitly set to 512 to match collection
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
    """Filter out legacy omni.isaac content and prioritize 5.0 API chunks."""
    
    modern_chunks = []

    for chunk in chunks:
        text = chunk["text"]

        if "omni.isaac" in text:
            # Exclude legacy chunks completely
            continue
        else:
            # Everything else is modern
            chunk["boosted_score"] = chunk.get("score", 0)
            modern_chunks.append(chunk)
    
    return modern_chunks




def rerank_chunks(query: str, chunks: List[Dict]) -> List[Dict]:
    """Rerank chunks with heavy preference for Isaac Sim 5.0 / Isaac Lab content."""
    
    keywords = set(query.lower().split())
    modern_indicators = ["isaacsim", "isaaclab"]  # simplified
    built_in_classes = ["jetbot", "differentialcontroller"]

    for chunk in chunks:
        text_lower = chunk["text"].lower()
        
        # Start with boosted_score from filtering
        score = chunk.get("boosted_score", 0)
        
        # Add keyword matches from query
        keyword_matches = sum(1 for kw in keywords if kw in text_lower)
        score += keyword_matches * 0.15
        
        # Boost chunks mentioning modern APIs
        modern_count = sum(1 for indicator in modern_indicators if indicator in text_lower)
        score += modern_count * 0.5

        # Extra boost for code examples
        if "```python" in text_lower or "import " in text_lower:
            score += 0.2
        
        # Extra boost for JetBot / DifferentialController mentions
        builtin_count = sum(1 for cls in built_in_classes if cls in text_lower)
        score += builtin_count * 0.7

        chunk["final_score"] = score

    # Sort by final score descending
    chunks.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    return chunks




def build_prompt(query: str, reranked_chunks: List[Dict]) -> str:
    """Build a structured prompt with context, excluding legacy chunks entirely."""
    
    # Only include modern chunks
    modern_chunks = reranked_chunks[:7]  # top 7 only

    if not modern_chunks:
        return (
            "You are an expert Isaac Sim 5.0 and Isaac Lab coding assistant.\n"
            "No relevant 5.0 documentation was found.\n"
            "Please respond with: 'I don't have enough context to answer this question accurately.'\n\n"
            f"User question: {query}"
        )

    context_parts = []
    for i, chunk in enumerate(modern_chunks, 1):
        score = chunk.get("final_score", chunk.get("boosted_score", 0))
        context_parts.append(f"[Context {i}] (relevance: {score:.2f})\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    # Updated migration / class map
    migration_map = """
ISAAC SIM 4.x → 5.0 API MIGRATION GUIDE:

OLD (4.x - DEPRECATED):          NEW (5.0 - USE THIS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from omni.isaac.core            → from isaacsim.core.api
from omni.isaac.core.world      → from isaacsim.core.api.world
from omni.isaac.core.objects    → from isaacsim.core.api.objects
from omni.isaac.wheeled_robots  → from isaacsim.robots.wheeled
from omni.isaac.core.utils      → from isaacsim.core.utils
World()                         → World() (same usage)
DynamicCuboid()                 → DynamicCuboid() (same usage)
WheeledRobot()                  → WheeledRobot() (new location)

BUILT-IN CLASSES TO USE IN 5.0:
JetBot()                 → JetBot() (built-in)
DifferentialController() → DifferentialController() (built-in)
"""

    return f"""You are an expert Isaac Sim 5.0 and Isaac Lab coding assistant.

{migration_map}

⚠️ CRITICAL RULES:
- Ignore any legacy 4.x omni.isaac imports in the context.
- Translate all "omni.isaac.*" imports to 5.0 equivalents.
- Only return Isaac Sim 5.0 / Isaac Lab imports in your code.
- Prefer using built-in classes like JetBot and DifferentialController when mentioned in query.

CONTEXT FROM DOCUMENTATION (5.0 only):
{context}

USER QUESTION:
{query}

RESPONSE (provide complete Isaac Sim 5.0 Python code):
"""



def answer_question(query: str, model_choice: str) -> tuple[str, str]:
    """Generate answer and return both code and debug info."""
    
    # Retrieve chunks
    chunks = retrieve_chunks(query, top_k=TOP_K)
    
    # Debug info
    debug_info = f"🔍 Retrieved {len(chunks)} chunks\n"
    debug_info += "Initial scores: " + ", ".join(f"{c['score']:.3f}" for c in chunks[:5]) + "\n\n"
    
    # Filter legacy content
    filtered_chunks = filter_chunks(chunks)
    debug_info += f"✅ After filtering: {len(filtered_chunks)} chunks\n"
    
    # Count modern vs legacy
    modern_count = sum(1 for c in filtered_chunks if "omni.isaac" not in c['text'])
    legacy_count = sum(1 for c in filtered_chunks if "omni.isaac" in c['text'])

    debug_info += f"   Modern API chunks: {modern_count}\n"
    debug_info += f"   Legacy API chunks: {legacy_count}\n\n"
    
    # Rerank
    reranked_chunks = rerank_chunks(query, filtered_chunks)
    
    # Show top 5 after reranking
    debug_info += "📊 Top 5 chunks after reranking:\n"
    for i, chunk in enumerate(reranked_chunks[:5], 1):
        snippet = chunk['text'][:100].replace('\n', ' ')
        
        # Mark legacy vs modern
        modern_mark = "✗" if "omni.isaac" in chunk['text'] else "✓"
        
        debug_info += f"  {i}. [{modern_mark}] Score: {chunk.get('final_score', 0):.3f} - {snippet}...\n"

    
    # Build prompt
    prompt = build_prompt(query, reranked_chunks)
    
    # Generate response
    try:
        response = ollama_client.generate(
            model=model_choice,
            prompt=prompt,
            options={
                "num_predict": 2048,
                "temperature": 0.2,
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
                choices=["qwen2.5-coder:7b-instruct", "qwen3-coder:latest"],
                value="qwen2.5-coder:7b-instruct",
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