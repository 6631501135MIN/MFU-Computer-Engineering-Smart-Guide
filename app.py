import re
import html
import faiss
import torch
import joblib
import streamlit as st
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================
# 1. Load Trained RAG Model and Scaler
# ==========================================

rag_model = joblib.load("best_ce_rag_model.pkl")
scaler = joblib.load("scaler.pkl")

chunks = rag_model["chunks"]
chunk_sources = rag_model["chunk_sources"]
knowledge_cards = rag_model["knowledge_cards"]

vectorstore = faiss.deserialize_index(rag_model["faiss_index_bytes"])

embedding_model_name = scaler["embedding_model_name"]
llm_model_name = scaler["llm_model_name"]

# ==========================================
# 2. Project Information
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
# 3. Load Models
# ==========================================

def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        embedding_model_name,
        device=device
    )

    return model, device


def load_llm_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
    llm_model = llm_model.to(device)

    return tokenizer, llm_model, device


embedding_model, device = load_embedding_model()
tokenizer, llm_model, device = load_llm_model()

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


def semantic_card_answer(question):
    context, retrieved_chunks, retrieved_sources, retrieved_scores = retrieve_context(question, top_k=1)

    if len(retrieved_chunks) == 0:
        return None

    best_chunk = retrieved_chunks[0]
    best_source = retrieved_sources[0]
    best_score = retrieved_scores[0]

    if best_score >= 0.50 and "Knowledge Card" in best_source and "Answer:" in best_chunk:
        answer = best_chunk.split("Answer:", 1)[1].strip()
        return answer

    return None


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


def get_answer(question):
    answer = keyword_card_answer(question)

    if answer is None:
        answer = semantic_card_answer(question)

    if answer is None:
        context, retrieved_chunks, retrieved_sources, retrieved_scores = retrieve_context(question, top_k=4)
        answer = generate_answer(question, context)

        if len(answer.strip()) < 10:
            answer = "The information is not available in the provided document."

    return answer


quick_questions = {
    "Program Overview": "What is the Computer Engineering program about?",
    "Degree Title": "What is the degree title?",
    "Learning Tracks": "What are the three learning tracks in Computer Engineering?",
    "Career Opportunities": "What career opportunities are available after graduation?",
    "Tuition Fee": "How much is the total program fee?",
    "Curriculum Structure": "What is the curriculum structure?",
    "PLOs": "What are the Program Learning Outcomes?",
    "Contact Information": "What is the contact information?"
}

# ==========================================
# 5. Streamlit App
# ==========================================

def run_streamlit_app():
    st.set_page_config(
        page_title=project_name,
        page_icon="🎓",
        layout="wide"
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #eef5ff 0%, #f8fbff 45%, #ffffff 100%);
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }

        .hero {
            background: linear-gradient(135deg, #0f3460 0%, #1f6fb2 100%);
            padding: 26px 32px;
            border-radius: 22px;
            color: white;
            margin-bottom: 22px;
            box-shadow: 0 14px 35px rgba(15, 52, 96, 0.20);
        }

        .hero h1 {
            font-size: 34px;
            margin-bottom: 8px;
            color: white;
            font-weight: 800;
        }

        .hero p {
            font-size: 17px;
            line-height: 1.5;
            color: #eef6ff;
            margin-bottom: 0px;
        }

        .left-card {
            background: white;
            padding: 20px;
            border-radius: 20px;
            border: 1px solid #d7e4f4;
            box-shadow: 0 10px 26px rgba(31, 111, 178, 0.08);
            color: #1f2937;
        }

        .left-card h3 {
            color: #0f3460;
            margin-top: 0px;
            font-size: 21px;
        }

        .left-card h4 {
            color: #0f3460;
            margin-bottom: 8px;
        }

        .left-card p {
            margin-bottom: 8px;
            font-size: 15px;
            color: #1f2937;
        }

        .quick-title {
            font-size: 19px;
            font-weight: 800;
            color: #0f3460;
            margin-top: 16px;
            margin-bottom: 10px;
        }

        div.stButton > button {
            border-radius: 13px;
            font-weight: 700;
            border: 1px solid #c9d9ec;
            background-color: #ffffff;
            color: #0f3460;
            min-height: 42px;
        }

        div.stButton > button:hover {
            background-color: #e8f2ff;
            border-color: #1f6fb2;
            color: #0f3460;
        }

        .chat-title {
            font-size: 24px;
            font-weight: 800;
            color: #0f3460;
            margin-bottom: 14px;
        }

        .chat-message-user {
            background: #e8f2ff;
            color: #1f2937;
            padding: 14px 17px;
            border-radius: 16px;
            margin-bottom: 10px;
            border: 1px solid #d7e4f4;
        }

        .chat-message-bot {
            background: #ffffff;
            color: #1f2937;
            padding: 14px 17px;
            border-radius: 16px;
            margin-bottom: 12px;
            border: 1px solid #d7e4f4;
            box-shadow: 0 6px 16px rgba(31, 111, 178, 0.07);
        }

        .chat-label {
            font-weight: 800;
            color: #0f3460;
            margin-bottom: 5px;
        }

        .stTextArea textarea {
            border-radius: 14px;
            border: 1px solid #c9d9ec;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "current_question" not in st.session_state:
        st.session_state["current_question"] = ""

    st.markdown(
        """
        <div class="hero">
            <h1>🎓 MFU Computer Engineering Smart Guide</h1>
            <p>
                Ask questions about the Computer Engineering Program at Mae Fah Luang University.
                The system retrieves answers from the provided PDF dataset using a RAG model.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    left_col, right_col = st.columns([1, 2.1], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="left-card">
                <h3>📌 Project Information</h3>
                <p><b>Group No.:</b> BDA_Project2_14</p>
                <p><b>Project Name:</b><br>MFU Computer Engineering Smart Guide</p>
                <h4>Group Members</h4>
                <p>6631501135 — Min Maung Maung</p>
                <p>6631501134 — Kyaw Thura</p>
                <p>6631501186 — Kyaw Phone Thu</p>
                <p>6631501183 — Phone Wai Yan Thaing</p>
                <p style="color:#64748b; font-size:14px;">
                    Quick questions stay available during the whole conversation.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="quick-title">Quick Questions</div>', unsafe_allow_html=True)

        for label, question_text in quick_questions.items():
            if st.button(label, use_container_width=True):
                answer = get_answer(question_text)

                st.session_state["messages"].append({
                    "role": "user",
                    "content": question_text
                })

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer
                })

                st.session_state["current_question"] = ""
                st.rerun()

        st.markdown("")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["current_question"] = ""
            st.rerun()

    with right_col:
        with st.container(border=True):
            st.markdown('<div class="chat-title">💬 Chat with the Smart Guide</div>', unsafe_allow_html=True)

            if len(st.session_state["messages"]) == 0:
                st.info("Choose a quick question or type your own question below.")

            for message in st.session_state["messages"]:
                safe_content = html.escape(message["content"]).replace("\n", "<br>")

                if message["role"] == "user":
                    st.markdown(
                        f"""
                        <div class="chat-message-user">
                            <div class="chat-label">You</div>
                            {safe_content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="chat-message-bot">
                            <div class="chat-label">Smart Guide</div>
                            {safe_content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            question = st.text_area(
                "Your Question",
                value=st.session_state["current_question"],
                height=95,
                placeholder="Ask about the Computer Engineering program..."
            )

            if st.button("Send", type="primary", use_container_width=True):
                if question.strip() == "":
                    st.warning("Please enter a question first.")
                else:
                    with st.spinner("Searching the knowledge base and generating answer..."):
                        answer = get_answer(question)

                    st.session_state["messages"].append({
                        "role": "user",
                        "content": question
                    })

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer
                    })

                    st.session_state["current_question"] = ""
                    st.rerun()

# ==========================================
# 6. Gradio App
# ==========================================

def gradio_chat(message, history):
    if message.strip() == "":
        return "Please enter a question."

    return get_answer(message)


def run_gradio_app(share=True, inline=True, prevent_thread_lock=True):
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #eef5ff 0%, #f8fbff 45%, #ffffff 100%) !important;
        font-family: Arial, sans-serif !important;
    }

    #hero {
        background: linear-gradient(135deg, #0f3460 0%, #1f6fb2 100%);
        padding: 26px 30px;
        border-radius: 22px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 14px 35px rgba(15, 52, 96, 0.20);
    }

    #hero h1 {
        margin: 0 0 8px 0;
        color: white;
        font-size: 32px;
    }

    #hero p {
        margin: 0;
        color: #eef6ff;
        font-size: 16px;
    }

    #left-panel {
        background: white;
        border: 1px solid #d7e4f4;
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 10px 26px rgba(31, 111, 178, 0.08);
    }

    #left-panel h3, #left-panel h4, #left-panel p {
        color: #1f2937 !important;
    }

    #left-panel h3, #left-panel h4 {
        color: #0f3460 !important;
    }

    #chatbot {
        border-radius: 20px !important;
        border: 1px solid #d7e4f4 !important;
        box-shadow: 0 10px 26px rgba(31, 111, 178, 0.08);
    }

    .quick-btn button {
        width: 100% !important;
        border-radius: 12px !important;
        background: white !important;
        border: 1px solid #c9d9ec !important;
        color: #0f3460 !important;
        font-weight: 700 !important;
    }
    """

    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        chat_state = gr.State([])

        gr.HTML(
            """
            <div id="hero">
                <h1>🎓 MFU Computer Engineering Smart Guide</h1>
                <p>
                    Ask questions about the Computer Engineering Program at Mae Fah Luang University.
                    Quick questions stay available while chatting.
                </p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=300, elem_id="left-panel"):
                gr.HTML(
                    """
                    <h3>📌 Project Information</h3>
                    <p><b>Group No.:</b> BDA_Project2_14</p>
                    <p><b>Project Name:</b><br>MFU Computer Engineering Smart Guide</p>
                    <h4>Group Members</h4>
                    <p>6631501135 — Min Maung Maung</p>
                    <p>6631501134 — Kyaw Thura</p>
                    <p>6631501186 — Kyaw Phone Thu</p>
                    <p>6631501183 — Phone Wai Yan Thaing</p>
                    <h4>Quick Questions</h4>
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

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=620,
                    type="messages",
                    show_copy_button=True,
                    elem_id="chatbot"
                )

                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask about the Computer Engineering program...",
                    lines=2
                )

                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")

        def add_message(message, history):
            if message.strip() == "":
                return history, history, ""

            history = history or []

            history.append({
                "role": "user",
                "content": message
            })

            answer = get_answer(message)

            history.append({
                "role": "assistant",
                "content": answer
            })

            return history, history, ""

        def quick_message(history, question):
            return add_message(question, history)

        def clear_messages():
            return [], [], ""

        send_btn.click(
            fn=add_message,
            inputs=[msg, chat_state],
            outputs=[chatbot, chat_state, msg]
        )

        msg.submit(
            fn=add_message,
            inputs=[msg, chat_state],
            outputs=[chatbot, chat_state, msg]
        )

        clear_btn.click(
            fn=clear_messages,
            inputs=[],
            outputs=[chatbot, chat_state, msg]
        )

        q1.click(lambda history: quick_message(history, quick_questions["Program Overview"]), inputs=[chat_state], outputs=[chatbot, chat_state, msg])
        q2.click(lambda history: quick_message(history, quick_questions["Degree Title"]), inputs=[chat_state], outputs=[chatbot, chat_state, msg])
        q3.click(lambda history: quick_message(history, quick_questions["Learning Tracks"]), inputs=[chat_state], outputs=[chatbot, chat_state, msg])
        q4.click(lambda history: quick_message(history, quick_questions["Career Opportunities"]), inputs=[chat_state], outputs=[chatbot, chat_state, msg])
        q5.click(lambda history: quick_message(history, quick_questions["Tuition Fee"]), inputs=[chat_state], outputs=[chatbot, chat_state, msg])
        q6.click(lambda history: quick_message(history, quick_questions["Curriculum Structure"]), inputs=[chat_state], outputs=[chatbot, chat_state, msg])
        q7.click(lambda history: quick_message(history, quick_questions["PLOs"]), inputs=[chat_state], outputs=[chatbot, chat_state, msg])
        q8.click(lambda history: quick_message(history, quick_questions["Contact Information"]), inputs=[chat_state], outputs=[chatbot, chat_state, msg])

    demo.launch(
        share=share,
        inline=inline,
        prevent_thread_lock=prevent_thread_lock
    )

# ==========================================
# 7. Run Correct UI
# ==========================================

def is_running_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if is_running_streamlit():
    run_streamlit_app()
else:
    if __name__ == "__main__":
        run_gradio_app()
