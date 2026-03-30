# AI-Powered Resume Screening System with RAG Pipeline

An intelligent resume screening system that automatically matches resumes to job descriptions using a RAG (Retrieval-Augmented Generation) pipeline. The system scores each candidate with a match percentage, identifies skill gaps, and provides hire/reject recommendations — all powered by free, open-source tools.

---

## Project Summary

This project was built to automate the resume screening process using AI. The system allows HR teams to upload multiple resumes and a job description, then instantly get a ranked list of candidates with detailed analysis for each one.

The core pipeline works in three stages:
- **Ingestion** — Resumes (PDF, DOCX, TXT) are parsed and split into chunks using LangChain's text splitter
- **Embedding & Storage** — Each chunk is converted into a vector using the `all-MiniLM-L6-v2` sentence transformer model and stored in Pinecone vector database for semantic search
- **Scoring** — Groq's LLaMA 3.3-70B model compares each resume against the job description and returns a structured JSON response with match score, matching skills, missing skills, experience match, and a final recommendation

The results are displayed in an interactive Streamlit dashboard with ranked candidate tables, bar charts, radar charts, gauge charts, and a CSV export feature.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| LLM | Groq API (LLaMA 3.3-70B) |
| Vector Database | Pinecone |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Text Splitting | LangChain Text Splitters |
| Document Parsing | PyPDF2, python-docx |
| Visualizations | Plotly |
| Language | Python 3.14 |
| Version Control | Git / GitHub |

---


### Get your free API keys
- **Groq** (free) → [console.groq.com](https://console.groq.com) → API Keys → Create API Key
- **Pinecone** (free tier) → [app.pinecone.io](https://app.pinecone.io) → API Keys → Copy key

### Run the app
```bash
python -m streamlit run app.py
```