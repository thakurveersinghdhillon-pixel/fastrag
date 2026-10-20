# FastRAG Phase 9.3: External Model Comparison Protocol

**Document Version**: `1.0.0`  
**Date**: October 16, 2026  
**Status**: Formal Research Protocol  
**Project**: FastRAG (`/Users/thakurveersingh/Documents/fastrag`)  

---

## 1. Research Objectives & Scope

The purpose of this protocol is to define an empirical, reproducible methodology for comparing **FastRAG** against external LLM baselines (such as Google Gemini and OpenAI / ChatGPT models).

### Core Research Questions:
1. **RAG vs. Parametric Memory**: How much do retrieval and CrossEncoder reranking improve factual answer accuracy, completeness, and citation traceability compared to a direct LLM answering from parametric weights alone?
2. **Abstention on Out-of-Corpus Questions**: Does FastRAG effectively prevent hallucinations by signaling lack of evidence on unanswerable questions, whereas direct LLMs generate speculative or fabricated answers?
3. **Citation Precision & Traceability**: Can external models generate accurate citation references without retrieval augmentation?

---

## 2. Dataset Eligibility & Split Isolation

To prevent empirical overfitting and ensure scientific rigor:
1. **Evaluation Split**:
   - Primary comparison is conducted exclusively on the **Development Split** (`datasets/v2/queries_dev.jsonl`).
   - The **Held-Out Test Split** (`datasets/v2/queries_test.jsonl`) remains strictly quarantined until all prompts, models, and protocols are finalized.
2. **Provenance Filtering**:
   - Only records with `annotation_provenance.status == "human_reviewed"` are eligible for quantitative claims.
   - For the Dev split, exactly **35 queries** meet this eligibility standard:
     - 33 **Answerable Queries** across `factual_direct`, `technical_identifiers`, `citation_grounding`, `multi_hop`, and `ambiguous_underspecified`.
     - 2 **Unanswerable / Negative Queries** (`unanswerable_negative`).
   - 25 unverified queries (`heuristic_candidate`, `pending_review`) in the Dev split are excluded from primary headline metrics.
3. **Audit of "Human-Reviewed" Annotations**:
   - The 35 eligible queries were authored and verified by the FastRAG Research Team against the canonical 36-document corpus.
   - Limitation: Because authors were project contributors rather than third-party double-blind annotators, potential author bias exists. External replication should validate these records with independent judges.

---

## 3. Compared Systems & Configurations

Every evaluated system receives the exact same set of 35 eligible queries with identical answerability ground truth:

### System A: FastRAG (RAG Mode)
- **Pipeline**: Two-Stage Retrieval Funnel (Hybrid BM25 + Dense vector with RRF fusion, top-20 candidates) $\to$ CrossEncoder reranking (`ms-marco-MiniLM-L-6-v2`, top-10 passages) $\to$ Context Assembly with token budgeting and `[Doc-1]` citation markers $\to$ LLM generation.
- **System Prompt**: `DEFAULT_SYSTEM_PROMPT` (`backend/app/generation/prompts.py`).
- **Sampling Parameters**: `temperature=0.0`, `max_tokens=1024`.

### System B: Direct LLM Baseline (Closed-Book Parametric QA)
- **Pipeline**: Zero retrieval. The query is submitted directly to the LLM without context passages.
- **System Prompt**:
  ```text
  You are a technical question-answering assistant.
  Answer the user's question accurately and concisely based on your general technical knowledge.
  If you do not know the answer or lack sufficient knowledge to answer accurately, explicitly state:
  "I do not have sufficient information to answer this question." Do not fabricate facts.
  ```
- **Sampling Parameters**: `temperature=0.0`, `max_tokens=1024`.
- **Purpose**: Isolates the factual contribution of the retrieval pipeline from the base model's internal memorization.

### System C: External Model Baselines (OpenAI / ChatGPT, Gemini)
- **Gemini**: `gemini-3.8-flash` via Google Generative Language OpenAI-compatible endpoint.
- **OpenAI / ChatGPT**: `gpt-4o` or `gpt-4o-mini` via OpenAI API endpoint (`https://api.openai.com/v1`).
- **Comparison Modes**: Both Direct (closed-book) and RAG (retrieval-augmented).

---

## 4. Evaluation Metrics & Deterministic Scoring

All primary scores are computed deterministically without relying on non-deterministic LLM-as-a-judge scoring:

1. **Token-Level F1, Precision, and Recall**:
   - Text is normalized (lower-cased, punctuation stripped, articles removed, whitespace normalized).
   - Precision = $|P \cap G| / |P|$, Recall = $|P \cap G| / |G|$, F1 = $2 \cdot P \cdot R / (P + R)$.
2. **Answer Completeness**:
   - Fraction of expected evidence concepts from `expected_evidence` covered in the response.
3. **Context Groundedness (Faithfulness)**:
   - Fraction of generated sentences whose claims are attested by the retrieved context block.
   - For direct (non-RAG) baselines, context groundedness is 0.0 by definition (no context provided).
4. **Citation Precision & Recall**:
   - Citation Recall: Fraction of target `source_doc_ids` cited in the answer text.
   - Citation Precision: Fraction of cited document IDs that are genuinely relevant ($\ge 1$).
   - Direct LLM baselines that lack access to document identifiers cannot produce grounded citation tags.
5. **Unanswerable Abstention Accuracy**:
   - On `unanswerable_negative` queries ($N=2$), measures whether the system explicitly signals lack of information instead of hallucinating:
     - 1.0 = Successfully abstains (e.g., *"does not contain sufficient information"*, *"cannot be answered"*).
     - 0.0 = Fabricates an answer to the unanswerable query.

---

## 5. Bounded Retries & Rate-Limit Handling

External API calls must handle transient HTTP 429 (rate limit) and HTTP 503 (service unavailable) responses:
- **Maximum Retries**: 3 attempts per query.
- **Backoff Strategy**: Exponential backoff with jitter: $T_{\text{wait}} = 2^{\text{attempt}} + \text{uniform}(0, 1)$ seconds.
- **Timeout**: Strictly bounded at 30 seconds per request.
- **Failure Transparency**: If an API call fails after 3 retries, the failure is recorded explicitly with error message and status code. **Never silently substitute mock output for a real API failure.**

---

## 6. ChatGPT Evaluation Methodology & Limitations

A major research goal is comparing FastRAG against ChatGPT. We establish the following standards:

1. **Consumer ChatGPT Web UI Limitation**:
   - The interactive ChatGPT consumer web interface (`chatgpt.com` / `chat.openai.com`) cannot serve as a scientific research baseline because:
     - It has no versioned API endpoint or deterministic temperature control.
     - Backend system instructions, hidden tool integrations, and model weights are modified continuously by OpenAI without notice.
     - Prompts cannot be executed in reproducible batch scripts.
2. **Reproducible ChatGPT Protocol**:
   - To benchmark "ChatGPT" fairly and reproducibly, comparisons must use the official OpenAI API targeting fixed, versioned snapshot models:
     - Model ID: `gpt-4o` (or `gpt-4o-mini`).
     - Base URL: `https://api.openai.com/v1`.
     - Parameters: `temperature=0.0`, identical prompt text, and identical question ordering.
3. **Current Environment Access Disclosure**:
   - In this development environment:
     - `OPENAI_API_KEY` is not configured.
     - The local execution sandbox isolates outgoing DNS and network traffic.
   - **Protocol Requirement**: Unless live API credentials and external network access are present, no ChatGPT or Gemini numbers may be fabricated. Provider unavailability must be stated as an empirical limitation.
