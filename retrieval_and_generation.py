# ✅ Bio_ClinicalBERT + Together AI with Structured Medical Prompt and Input Gate

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from dotenv import load_dotenv
import os
from together import Together
from sklearn.metrics.pairwise import cosine_similarity
import re

# === Load API Key ===
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

# === Load Bio_ClinicalBERT model ===
print("🧠 Loading Bio_ClinicalBERT...")
word_embedding_model = models.Transformer("emilyalsentzer/Bio_ClinicalBERT", max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
embed_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
print("✅ Query embedding shape:", embed_model.encode(["test"]).shape)

# === Load dataset and embeddings ===
print("📄 Loading dataset and embeddings...")
df = pd.read_csv("data/full_dataset_cleaned.csv")
embedding_matrix = np.load("embeddings/embeddings.npy")
print("✅ Embeddings loaded with shape:", embedding_matrix.shape)

# === Load FAISS index ===
faiss_index = faiss.read_index("embeddings/faiss_index.bin")

# === Get user query ===
user_query = input("\n📝 Enter patient case summary: ").strip()

# === Check if query is a valid clinical sentence (uses patterns of structure) ===
def is_structured_clinical_summary(text):
    structured_keywords = ["presents", "presented", "complains of", "history of", "with"]
    return bool(re.search(r"\b\d{2,3}-year-old\b", text, re.IGNORECASE)) and any(kw in text.lower() for kw in structured_keywords)

if not is_structured_clinical_summary(user_query):
    print("⚠️ Rejected: This is not a structured clinical case. Please enter a detailed patient summary (e.g., '60-year-old male presents with chest pain and shortness of breath').")
    exit()

# === Biological contradiction gate (before retrieval or encoding) ===
biological_contradictions = [
    ("male", "pregnancy"), ("male", "positive hcg"), ("male", "pregnant"),
    ("female", "testicular"), ("female", "prostate")
]
query_lower = user_query.lower()
invalid_bio_case = any(gender_term in query_lower and contradiction in query_lower for gender_term, contradiction in biological_contradictions)

# === Structured, guarded clinical prompt ===
def generate_prompt(user_query, retrieved_text):
    return f"""
You are a highly trained clinical decision support AI assistant for ICU physicians. You only respond to structured medical case summaries that describe real patients with symptoms, clinical history, lab results, or physical findings.

If the input contains biologically contradictory information (e.g., a male patient reporting pregnancy or a female patient presenting with testicular symptoms), do not generate any answer.

---

**PATIENT CASE:**
{user_query}

**SIMILAR CASES FOR REFERENCE:**
{retrieved_text}

Now, based on the above case, provide:

1. **DIFFERENTIAL DIAGNOSIS:**
   - Primary diagnosis (most likely)
   - Alternative diagnoses to consider
   - Key distinguishing features

2. **RECOMMENDED DIAGNOSTIC TESTS:**
   - Labs, imaging, and rationale

3. **TREATMENT PLAN:**
   - Medications, supportive care, monitoring

4. **LIFESTYLE & EDUCATION:**
   - Patient counseling and prevention

5. **CLINICAL ASSESSMENT (One-paragraph summary):**
   - Summarize the case, likely diagnosis, and your overall plan.
"""

if invalid_bio_case:
    print("\n🚫 Biologically implausible input detected. Skipping cosine similarity and generation.")
    print("\n================================================================================")
    print("COMPREHENSIVE CLINICAL RECOMMENDATION")
    print("================================================================================")
    print("> ⚠️ The provided case contains biologically implausible or contradictory information (e.g., gender-inconsistent test result or symptom). Please verify and revise the input accordingly.")
    print("\n⚠️ Cosine similarity was skipped due to biologically invalid input.")
    print("\n⚠️  DISCLAIMER: This is an AI-generated recommendation for educational purposes only.")
else:
    query_embedding = embed_model.encode([user_query]).astype("float32")
    print("\n🔍 Query shape:", query_embedding.shape)

    k_initial = 3
    _, initial_indices = faiss_index.search(query_embedding, k=k_initial)
    initial_embeddings = embedding_matrix[initial_indices[0]]
    similarities = cosine_similarity(query_embedding, initial_embeddings)[0]
    sorted_idx = np.argsort(similarities)[::-1]
    best_indices = [initial_indices[0][i] for i in sorted_idx[:2]]
    retrieved_notes = [df["input"].iloc[i] for i in best_indices]
    retrieved_text = "\n\n---\n\n".join(retrieved_notes)

    print("\n🔢 Cosine Similarity with retrieved notes:")
    for rank, i in enumerate(sorted_idx, 1):
        print(f"  Candidate {rank}: FAISS Index {initial_indices[0][i]}, Cosine = {similarities[i]:.4f}")

    # === Call Together AI ===
    print("\n🚀 Generating clinical recommendation using LLaMA-3 from Together AI...")
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": generate_prompt(user_query, retrieved_text)}
        ],
        max_tokens=1500,
        temperature=0.3,
        top_p=0.9
    )

    response_text = response.choices[0].message.content.strip()

    print("\n================================================================================")
    print("COMPREHENSIVE CLINICAL RECOMMENDATION")
    print("================================================================================")
    print(response_text)
    print("================================================================================")

    print("\n⚠️  DISCLAIMER: This is an AI-generated recommendation for educational purposes only.")
    print("   Always consult qualified healthcare professionals for actual medical care.")
