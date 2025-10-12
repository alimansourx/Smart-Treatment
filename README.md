# Smart Treatment: AI-Driven Personalized Medical Care

### Team Name: Smart Treatment Team

**Institution:** Egypt-Japan University of Science and Technology
(E-JUST), Alexandria, Egypt

## Overview

Smart Treatment is an AI-driven clinical decision support system
designed to assist healthcare professionals in generating personalized,
evidence-based medical recommendations.\
By combining Retrieval-Augmented Generation (RAG) with biomedical
embeddings and large language models, the system retrieves semantically
similar patient cases and produces comprehensive diagnostic, treatment,
and lifestyle recommendations.

## Features

-   Structured Patient Input: Accepts demographics, complaints,
    medications, and history.
-   Biomedical Embeddings: Uses BioClinicalBERT for semantic similarity.
-   Efficient Retrieval: FAISS-based vector search to identify relevant
    clinical cases.
-   Retrieval-Augmented Generation: Meta-LLaMA model (via Together AI
    API) generates clinical assessments.
-   Streamlit Interface: Simple, intuitive UI for clinicians to input
    cases and view results.
-   Privacy-Preserving: Operates on anonymized data (MIMIC-IV dataset).
-   Evaluation: Combines quantitative metrics (similarity scores,
    clustering) and qualitative human assessment.

## System Architecture

1.  Data Preprocessing: Cleans and structures clinical notes into JSONL
    format.
2.  Embedding Generation: Uses domain-tuned transformer models to embed
    cases.
3.  Vector Indexing: Stores embeddings in FAISS for high-speed
    retrieval.
4.  RAG Pipeline: Retrieves top-k relevant cases and prompts Meta-LLaMA
    to generate outputs.
5.  Streamlit Interface: Displays differential diagnoses, tests,
    treatment, and lifestyle recommendations.

## Setup and Installation

### 1. Clone the repository

    git clone https://github.com/alimansourx/smart-treatment.git
    cd smart-treatment

### 2. Create a virtual environment

    python -m venv venv
    source venv/bin/activate     # (Linux/Mac)
    venv\Scripts\activate        # (Windows)

### 3. Install dependencies

    pip install -r requirements.txt

### 4. Set up your environment variables

Create a `.env` file in the root directory and add your API key:

    TOGETHER_API_KEY=your_api_key_here

### 5. Run the application

    streamlit run app.py

The application will open in your browser at `http://localhost:8501`.

## Core Dependencies

  -----------------------------------------------------------------------
  Library                         Purpose
  ------------------------------- ---------------------------------------
  Python 3.11+                    Core language

  Streamlit                       User interface

  Transformers (Hugging Face)     Embedding and generation models

  FAISS                           Vector search and similarity indexing

  PyTorch                         Deep learning framework

  Sentence-Transformers           Biomedical embeddings

  Pandas / NumPy / Polars         Data manipulation

  Matplotlib / Plotly / Seaborn   Visualization

  Together API                    LLaMA model inference
  -----------------------------------------------------------------------

## Generative AI Usage Log

This project made ethical and transparent use of Generative AI tools
throughout development.\
Below are key instances of their integration.

  -----------------------------------------------------------------------------------------
  Task            Tool Used   Prompt Summary    AI Output / Assistance    Reflection
  --------------- ----------- ----------------- ------------------------- -----------------
  System          ChatGPT-4   "Design a modular Produced a high-level     Saved significant
  Architecture                architecture for  architecture (data        design time and
  Drafting                    an AI-driven      preprocessing, embedding, clarified modular
                              clinical decision retrieval, generation,    structure.
                              support system    UI).                      
                              using RAG."                                 

  Debugging FAISS ChatGPT-4   "Why does FAISS   Identified normalization  Prevented hours
  Retrieval                   return            issue and suggested using of manual
                              inconsistent      `faiss.normalize_L2()`.   debugging.
                              nearest                                     
                              neighbors?"                                 

  Prompt          ChatGPT-4   "Optimize prompt  Generated an improved     Improved clinical
  Engineering for             to generate       structured prompt         coherence and
  LLaMA                       structured        template for LLaMA        completeness by
                              medical           generation.               \~25%.
                              recommendations                             
                              (Diagnosis,                                 
                              Tests, Treatment,                           
                              Lifestyle)."                                

  Documentation & ChatGPT-4   "Write a concise  Provided initial text     Accelerated
  User Guide                  README section    refined into final        writing and
  Writing                     explaining setup  documentation.            improved
                              and features of a                           readability.
                              medical RAG app."                           

  Code            GitHub      Inline code       Generated working UI      Reduced
  Boilerplate for Copilot     completion while  components and validation implementation
  Streamlit UI                building form and snippets.                 time and ensured
                              display logic.                              consistent
                                                                          syntax.

  Research Paper  ChatGPT-4   "Refine and       Improved clarity,         Enhanced paper
  Abstract                    formalize this    coherence, and academic   quality for
  Polishing                   abstract for      tone.                     submission.
                              academic                                    
                              publication."                               
  -----------------------------------------------------------------------------------------

## Evaluation Summary

-   BioClinicalBERT achieved 88% semantic clustering accuracy,
    outperforming BioBERT and SapBERT.
-   Meta-LLaMA achieved 92% clinically valid and 87% complete outputs
    (expert review).
-   Retrieval-guided prompting improved diagnosis accuracy by 25% over
    zero-shot generation.

## Future Work

-   Integration with real-time Electronic Health Record (EHR) systems
-   Multilingual and multimodal data support
-   Deployment on secure hospital infrastructure
-   Development of a **smart health bracelet** for continuous health
    monitoring
-   Improved **Arabic language processing** for broader regional
    accessibility

## Team Members

-   Aly Hossam Mansour
-   Ahmed Hassan El-Sayed
-   Fairouz Maher Dakhly
-   Mohamed Said Mahmoud
-   Nada Mohamed Ahmed
-   Youssef Salah Mostafa

## License

**© 2025 Smart Treatment Team --- All Rights Reserved**\
This software and its documentation are proprietary and may not be
copied, distributed, or used without written permission from the
authors.

## Acknowledgment

We extend our gratitude to the E-JUST CSIT Department and the Center of
Excellence (COE) for their continuous guidance and support.\
Special thanks to all healthcare professionals and evaluators who
provided invaluable feedback on system usability and medical accuracy.


## Live Presentation

For a brief summary of the project, please use the link below.
https://drive.google.com/file/d/13ynxf0tC9FimjHJwSggKX2eNaNex-C0K/view?usp=sharing
