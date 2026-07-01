import gradio as gr
from pathlib import Path
from modules.semantic_indexer import SemanticIndex
from config import VECTOR_DIR, METADATA_PATH
from modules.utils import load_json

index = None
def load_index():
    global index
    if (VECTOR_DIR / "faiss.index").exists():
        index = SemanticIndex()
        index.load()
        return "Index loaded!"
    return "No index found. Process a video first."

def query(question, top_k):
    if index is None:
        return "Index not loaded."
    results = index.query(question, top_k=int(top_k))
    out = ""
    for r in results:
        ts = r.chunk.timestamp_s
        t = f"{int(ts//60):02d}:{int(ts%60):02d}" if ts else "??:??"
        out += f"[#{r.rank}] Score: {r.combined_score:.3f} | "
        out += f"Slide: {r.chunk.slide_index} | Time: {t}\n"
        out += f"{r.chunk.text[:300]}\n\n"
    return out

with gr.Blocks(title="Smart Lecturer") as demo:
    gr.Markdown("# Smart Lecturer\nSemantic search over lecture transcripts")
    with gr.Row():
        q_input = gr.Textbox(label="Your question", lines=2)
        topk = gr.Slider(2, 10, value=5, step=1, label="Results")
    btn = gr.Button("Search", variant="primary")
    output = gr.Textbox(label="Results", lines=15)
    btn.click(query, inputs=[q_input, topk], outputs=output)
    demo.load(load_index)

demo.launch()
