# ✅ Streamlit App: Updated to match retrieval_and_generation.py (with Input Gate & Bio Checks)

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, models
from dotenv import load_dotenv
import os
from together import Together
from sklearn.metrics.pairwise import cosine_similarity
import re

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Clinical Decision Support System",
    page_icon="\U0001F3E5",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS ===
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTextArea textarea { font-family: 'Courier New', monospace; }
    .stMarkdown { font-size: 1.1em; }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# === HEADER ===
st.title("\U0001F3E5 Clinical Decision Support System")
st.markdown("""
This AI-powered system assists healthcare professionals by providing structured, evidence-based clinical recommendations 
based on similar patient cases.
""")

# === SIDEBAR ===
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system uses:
    - Bio_ClinicalBERT for semantic understanding
    - FAISS for similarity search
    - LLaMA-3 (Together AI) for clinical reasoning
    """)
    st.header("Instructions")
    st.markdown("""
    1. Enter a structured patient case (e.g., age, gender, symptoms)
    2. Avoid biologically implausible contradictions
    3. Click 'Generate Analysis' to receive AI-powered recommendations
    """)

# === LOAD MODELS & DATA ===
if 'model_loaded' not in st.session_state:
    with st.spinner('Loading AI models...'):
        load_dotenv()
        api_key = os.getenv("TOGETHER_API_KEY")
        st.session_state.client = Together(api_key=api_key)

        word_embedding_model = models.Transformer("emilyalsentzer/Bio_ClinicalBERT", max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        st.session_state.embed_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        st.session_state.df = pd.read_csv("data/full_dataset_cleaned.csv")
        st.session_state.embedding_matrix = np.load("embeddings/embeddings.npy")
        st.session_state.faiss_index = faiss.read_index("embeddings/faiss_index.bin")

        st.session_state.model_loaded = True

# === INPUT ===
st.header("Patient Case Input")
user_query = st.text_area("Enter patient case summary:", height=200, placeholder="Example: 60-year-old male presents with chest pain and shortness of breath...")

# === STRUCTURE & BIOLOGICAL CHECKS ===
def is_structured_clinical_summary(text):
    pattern = r"\b\d{2,3}-year-old\b.*?(presents|presented|complains|has a history|was admitted|with|suffers from|diagnosed with)"
    return bool(re.search(pattern, text, re.IGNORECASE))

def has_biological_contradiction(text):
    contradictions = [
        ("male", "pregnancy"), ("male", "positive hcg"), ("male", "pregnant"),
        ("female", "testicular"), ("female", "prostate")
    ]
    text = text.lower()
    return any(g in text and c in text for g, c in contradictions)

# === BUTTON ===
if st.button("Generate Analysis"):
    if not user_query.strip():
        st.error("Please enter a patient case summary.")
    elif not is_structured_clinical_summary(user_query):
        st.error("⚠️ Please enter a structured case (e.g., '60-year-old male presents with...')")
    elif has_biological_contradiction(user_query):
        st.warning("🚫 Biologically implausible case detected. No generation performed.")
        st.markdown("""
            > ⚠️ The provided case includes gender-inconsistent findings (e.g., male with pregnancy).
            Please verify and revise the input accordingly.
        """)
    else:
        with st.spinner('Retrieving similar cases and generating recommendation...'):
            # === EMBEDDING + RETRIEVAL ===
            query_embedding = st.session_state.embed_model.encode([user_query]).astype("float32")
            k = 3
            _, idxs = st.session_state.faiss_index.search(query_embedding, k)
            retrieved_embeddings = st.session_state.embedding_matrix[idxs[0]]
            sims = cosine_similarity(query_embedding, retrieved_embeddings)[0]
            sorted_idx = np.argsort(sims)[::-1]
            top_idxs = [idxs[0][i] for i in sorted_idx[:2]]
            retrieved_notes = [st.session_state.df["input"].iloc[i] for i in top_idxs]
            retrieved_text = "\n\n---\n\n".join(retrieved_notes)

            # === PROMPT CONSTRUCTION ===
            def generate_prompt(query, context):
                return f"""
You are a highly trained clinical decision support AI assistant for ICU physicians. Only respond to structured patient summaries with symptoms, history, or findings.

---
**PATIENT CASE:**
{query}

**SIMILAR CASES FOR REFERENCE:**
{context}

Now, based on the above case, provide:

1. **DIFFERENTIAL DIAGNOSIS:**
   - Primary diagnosis (most likely)
   - Alternative diagnoses
   - Key distinguishing features

2. **RECOMMENDED DIAGNOSTIC TESTS:**
   - Labs, imaging, and rationale

3. **TREATMENT PLAN:**
   - Medications, supportive care

4. **LIFESTYLE & EDUCATION:**
   - Counseling, prevention tips

5. **CLINICAL ASSESSMENT:**
   - Summary paragraph
"""

            prompt = generate_prompt(user_query, retrieved_text)

            response = st.session_state.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3,
                top_p=0.9
            )

            result = response.choices[0].message.content.strip()

            # === OUTPUT ===
            st.markdown("## 🩺 Clinical Recommendation")
            st.markdown(result)

            st.markdown("""
            ---
            ⚠️ **Disclaimer**: This is an AI-generated recommendation for educational purposes only.
            Always consult qualified healthcare professionals for medical decisions.
            """)

# === FOOTER ===
st.markdown("""
---
<div style='text-align: center'>
    <p>Developed for healthcare professionals | Powered by Bio_ClinicalBERT & LLaMA-3</p>
</div>
""", unsafe_allow_html=True)
