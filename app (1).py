import re
import faiss
import torch
import joblib
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================
# 1. Load Trained RAG Model and Scaler
# ==========================================

rag_model = joblib.load("best_ce_rag_model.pkl")
scaler = joblib.load("scaler.pkl")

faiss_index_bytes = rag_model["faiss_index_bytes"]
chunks = rag_model["chunks"]
chunk_sources = rag_model["chunk_sources"]
knowledge_cards = rag_model["knowledge_cards"]

vectorstore = faiss.deserialize_index(faiss_index_bytes)

embedding_model_name = scaler["embedding_model_name"]
llm_model_name = scaler["llm_model_name"]

# ==========================================
# 2. Load Embedding Model and Local LLM
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = SentenceTransformer(
    embedding_model_name,
    device=device
)

tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
llm_model = llm_model.to(device)

# ==========================================
# 3. Project Information
# ==========================================

project_name = "MFU Computer Engineering Smart Guide"
group_no = "BDA_Project2_14"

group_members = [
    "6631501135 — Min Maung Maung",
    "6631501134 — Kyaw Thura",
    "6631501186 — Kyaw Phone Thu",
    "6631501183 — Phone Wai Yan Thaing"
]

# ==========================================
# 4. Helper Functions
# ==========================================

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_answer(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("Answer:", "")
    text = text.strip()
    return text

def keyword_card_answer(question):
    q = normalize_text(question)

    best_card = None
    best_score = 0

    for card in knowledge_cards:
        score = 0

        for keyword in card["keywords"]:
            keyword_clean = normalize_text(keyword)

            if keyword_clean in q:
                score += 3

            for word in keyword_clean.split():
                if len(word) > 2 and word in q:
                    score += 1

        if score > best_score:
            best_score = score
            best_card = card

    if best_card and best_score >= 2:
        return best_card["answer"]

    return None

# ==========================================
# 5. Retrieve Related Information
# ==========================================

def retrieve_context(question, top_k=4):
    question_embedding = embedding_model.encode(
        ["query: " + question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = vectorstore.search(question_embedding, top_k)

    retrieved_chunks = []
    retrieved_sources = []
    retrieved_scores = []

    for score, index in zip(scores[0], indices[0]):
        if index != -1:
            retrieved_chunks.append(chunks[index])
            retrieved_sources.append(chunk_sources[index])
            retrieved_scores.append(float(score))

    context = "\n\n".join(retrieved_chunks)

    return context, retrieved_chunks, retrieved_sources, retrieved_scores

# ==========================================
# 6. Direct Semantic Answer
# ==========================================

def semantic_card_answer(question):
    context, retrieved_chunks, retrieved_sources, retrieved_scores = retrieve_context(question, top_k=1)

    if len(retrieved_chunks) == 0:
        return None

    best_chunk = retrieved_chunks[0]
    best_source = retrieved_sources[0]
    best_score = retrieved_scores[0]

    if best_score >= 0.50 and best_source.startswith("Knowledge Card") and "Answer:" in best_chunk:
        answer = best_chunk.split("Answer:", 1)[1].strip()
        return answer

    return None

# ==========================================
# 7. Generate Answer from Local LLM
# ==========================================

def generate_answer(question, context):
    prompt = f"""
Use the context to answer the question.
Give only the final answer.
Keep it short and clear.
Use maximum 4 bullet points.
Do not repeat the whole context.
Do not use outside knowledge.
If the answer is not in the context, say: The information is not available in the provided document.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(device)

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=120,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = clean_answer(answer)

    return answer

# ==========================================
# 8. Main Chatbot Function
# ==========================================

def bot_answer(message, history):
    if message.strip() == "":
        return history, history, ""

    history = history or []

    history.append({
        "role": "user",
        "content": message
    })

    answer = keyword_card_answer(message)

    if answer is None:
        answer = semantic_card_answer(message)

    if answer is None:
        context, retrieved_chunks, retrieved_sources, retrieved_scores = retrieve_context(message, top_k=4)
        answer = generate_answer(message, context)

        if len(answer.strip()) < 10:
            answer = "The information is not available in the provided document."

    history.append({
        "role": "assistant",
        "content": answer
    })

    return history, history, ""

def quick_answer(history, question):
    return bot_answer(question, history)

def clear_chat():
    return [], [], ""

# ==========================================
# 9. Custom UI Style
# ==========================================

custom_css = """
* {
    box-sizing: border-box;
}

.gradio-container {
    background: #f4f7fb !important;
    font-family: Arial, sans-serif !important;
    color: #1f2937 !important;
}

#main-wrapper {
    max-width: 1280px;
    margin: 0 auto;
    padding: 14px;
}

#hero {
    background: linear-gradient(135deg, #123f72 0%, #2678ba 100%);
    border-radius: 18px;
    padding: 24px 30px;
    color: white !important;
    margin-bottom: 18px;
    box-shadow: 0 10px 26px rgba(18, 63, 114, 0.22);
}

#hero h1 {
    margin: 0 0 8px 0;
    font-size: 34px;
    font-weight: 800;
    color: white !important;
}

#hero p {
    margin: 0;
    font-size: 17px;
    color: #eef6ff !important;
    line-height: 1.5;
}

#left-panel {
    background: white !important;
    border: 1px solid #d7e3f1 !important;
    border-radius: 18px !important;
    padding: 18px !important;
    height: 690px;
    overflow-y: auto;
    box-shadow: 0 8px 22px rgba(38, 120, 186, 0.08);
}

#left-panel h3, 
#left-panel h4, 
#left-panel p, 
#left-panel b {
    color: #1f2937 !important;
}

#left-panel h3 {
    margin-top: 0;
    color: #123f72 !important;
}

#quick-title {
    color: #123f72 !important;
    font-weight: 800;
    margin-top: 18px;
    margin-bottom: 10px;
    font-size: 18px;
}

.quick-btn button {
    width: 100% !important;
    border-radius: 12px !important;
    background: #f8fbff !important;
    border: 1px solid #c9d9ec !important;
    color: #123f72 !important;
    font-weight: 700 !important;
    margin-bottom: 7px !important;
    min-height: 42px !important;
}

.quick-btn button:hover {
    background: #e7f1ff !important;
    border-color: #2678ba !important;
}

#chat-card {
    background: white !important;
    border: 1px solid #d7e3f1 !important;
    border-radius: 18px !important;
    padding: 14px !important;
    box-shadow: 0 8px 22px rgba(38, 120, 186, 0.08);
}

#chatbot {
    height: 560px !important;
    background: #ffffff !important;
    border-radius: 14px !important;
    border: 1px solid #d7e3f1 !important;
    overflow-y: auto !important;
}

.message {
    font-size: 15px !important;
    line-height: 1.55 !important;
}

#input-box textarea {
    border-radius: 12px !important;
    border: 1px solid #c9d9ec !important;
    color: #1f2937 !important;
    background: white !important;
}

.send-btn button {
    border-radius: 12px !important;
    background: #1f6fb2 !important;
    color: white !important;
    font-weight: 800 !important;
    min-height: 50px !important;
}

.clear-btn button {
    border-radius: 12px !important;
    background: #eef4fb !important;
    color: #123f72 !important;
    font-weight: 700 !important;
}

@media screen and (max-width: 900px) {
    #hero h1 {
        font-size: 26px;
    }

    #left-panel {
        height: auto;
        max-height: 460px;
    }

    #chatbot {
        height: 520px !important;
    }
}
"""

# ==========================================
# 10. Build Gradio UI
# ==========================================

with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
    chat_state = gr.State([])

    with gr.Column(elem_id="main-wrapper"):
        gr.HTML(
            """
            <div id="hero">
                <h1>🎓 MFU Computer Engineering Smart Guide</h1>
                <p>
                    Ask questions about the Computer Engineering Program at Mae Fah Luang University.
                    The assistant retrieves answers from the provided PDF dataset.
                </p>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=300, elem_id="left-panel"):
                gr.HTML(
                    """
                    <h3>📌 Project Information</h3>
                    <p><b>Group No.:</b> BDA_Project2_14</p>
                    <p><b>Project Name:</b><br>MFU Computer Engineering RAG Assistant</p>

                    <h4>Group Members</h4>
                    <p>6631501135 — Min Maung Maung</p>
                    <p>6631501134 — Kyaw Thura</p>
                    <p>6631501186 — Kyaw Phone Thu</p>
                    <p>6631501183 — Phone Wai Yan Thaing</p>

                    <div id="quick-title">Quick Questions</div>
                    """
                )

                q1 = gr.Button("Program Overview", elem_classes="quick-btn")
                q2 = gr.Button("Degree Title", elem_classes="quick-btn")
                q3 = gr.Button("Learning Tracks", elem_classes="quick-btn")
                q4 = gr.Button("Career Opportunities", elem_classes="quick-btn")
                q5 = gr.Button("Tuition Fee", elem_classes="quick-btn")
                q6 = gr.Button("Curriculum Structure", elem_classes="quick-btn")
                q7 = gr.Button("PLOs", elem_classes="quick-btn")
                q8 = gr.Button("Contact Information", elem_classes="quick-btn")

            with gr.Column(scale=2, min_width=500, elem_id="chat-card"):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    type="messages",
                    show_copy_button=True,
                    elem_id="chatbot"
                )

                with gr.Row():
                    message = gr.Textbox(
                        placeholder="Ask about the Computer Engineering program...",
                        label="Your Question",
                        lines=2,
                        scale=5,
                        elem_id="input-box"
                    )

                    send_btn = gr.Button("Send", scale=1, elem_classes="send-btn")

                clear_btn = gr.Button("Clear Chat", elem_classes="clear-btn")

        send_btn.click(
            fn=bot_answer,
            inputs=[message, chat_state],
            outputs=[chatbot, chat_state, message]
        )

        message.submit(
            fn=bot_answer,
            inputs=[message, chat_state],
            outputs=[chatbot, chat_state, message]
        )

        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, chat_state, message]
        )

        q1.click(lambda history: quick_answer(history, "What is the Computer Engineering program about?"), inputs=[chat_state], outputs=[chatbot, chat_state, message])
        q2.click(lambda history: quick_answer(history, "What is the degree title?"), inputs=[chat_state], outputs=[chatbot, chat_state, message])
        q3.click(lambda history: quick_answer(history, "What are the three learning tracks in Computer Engineering?"), inputs=[chat_state], outputs=[chatbot, chat_state, message])
        q4.click(lambda history: quick_answer(history, "What career opportunities are available after graduation?"), inputs=[chat_state], outputs=[chatbot, chat_state, message])
        q5.click(lambda history: quick_answer(history, "How much is the total program fee?"), inputs=[chat_state], outputs=[chatbot, chat_state, message])
        q6.click(lambda history: quick_answer(history, "What is the curriculum structure?"), inputs=[chat_state], outputs=[chatbot, chat_state, message])
        q7.click(lambda history: quick_answer(history, "What are the Program Learning Outcomes?"), inputs=[chat_state], outputs=[chatbot, chat_state, message])
        q8.click(lambda history: quick_answer(history, "What is the contact information?"), inputs=[chat_state], outputs=[chatbot, chat_state, message])

# ==========================================
# 11. Launch App
# ==========================================

if __name__ == "__main__":
    demo.launch()
