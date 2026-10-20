# FastRAG Phase 9.9: Generation Quality Evaluation Report

**Date**: October 18, 2026  
**Status**: Completed Audit & Pipeline Evaluation  
**Branch**: `main`  
**Scope**: Generation evaluation harness audit; metric classification and scoring logic; answerable vs. unanswerable query separation; reference answer and evidence span audit; development pipeline evaluation; and live LLM evaluation requirements.

---

## 1. Executive Summary

Phase 9.9 audited the FastRAG answer generation evaluation harness (`evaluation/generation_eval.py` and `scripts/run_generation_eval.py`) to determine whether FastRAG generates answers that are accurate, grounded in retrieved context, and appropriately abstain when context is insufficient.

### Key Findings:
1. **Clear Distinction Between Mock Pipeline Validation vs. Real Answer-Quality Evaluation**:
   - The repository includes a deterministic `MockLLMProvider` designed for zero-dependency offline pipeline verification. It generates template responses from the first retrieved chunk snippet.
   - Evaluating `MockLLMProvider` validates pipeline mechanics, context assembly budgeting, citation tag resolution, and threshold-based short-circuiting, but **does not measure genuine LLM semantic synthesis, reasoning, or real factual correctness**.
   - Genuine answer quality requires an actual LLM provider (`gemini-3.8-flash` or `gpt-4o-mini`).
2. **Ground Truth & Reference Answer Availability**:
   - Dataset v2 (`datasets/v2/queries_dev.jsonl`) contains complete, high-quality ground truth:
     - All 50 answerable dev queries contain curated `reference_answer`, `expected_evidence` spans, `source_doc_ids`, graded `relevance_judgments`, and `annotation_provenance`.
     - All 10 unanswerable queries explicitly specify `reference_answer: None`, `expected_evidence: []`, and `source_doc_ids: []`.
   - No ground-truth data is missing; no reference annotations needed to be fabricated.
3. **Answerable and Unanswerable Separation**:
   - Answerable and unanswerable queries are strictly segregated in `evaluation/generation_eval.py`:
     - Answerables are evaluated for Token F1, precision, recall, completeness, citation precision/recall, and false abstention rate.
     - Unanswerables are evaluated exclusively for unanswerable abstention accuracy.
4. **Offline Development Pipeline Evaluation**:
   - Executed full development evaluation ($N=60$ queries) with the Phase 9.4/9.5 pipeline configuration (`min_relevance_score = -2.0`, `ordering_strategy = "boundary"`, `blending_alpha = 0.5`).
   - Saved to [`benchmarks/results/generation_phase9_9_dev_verified_mock_2026-10-18T16-46-57_100736+00-00.json`](file:///Users/thakurveersingh/Documents/fastrag/benchmarks/results/generation_phase9_9_dev_verified_mock_2026-10-18T16-46-57_100736+00-00.json).
   - Unanswerable abstention accuracy: **80.0% (8/10)**.
   - False abstention rate on answerables: **12.0% (6/50)**.
5. **Live LLM Stopping Safeguard**:
   - In accordance with the prompt instructions (*"If real generation quality cannot be evaluated without a live LLM/API call, stop and report the required provider, model, and configuration. Do not make live API calls without approval."*), **zero live API calls were made**.
   - The required provider, model, configuration, and quota requirements are documented in Section 5.

---

## 2. Step 1: Audit of Existing Generation Evaluation Design

### 2.1 Current Generation Metrics and Scoring Logic
Implemented in [`evaluation/generation_eval.py`](file:///Users/thakurveersingh/Documents/fastrag/evaluation/generation_eval.py):

| Metric | Target Construct | Formula / Scoring Logic |
| :--- | :--- | :--- |
| **Token F1 / Prec / Rec** | Lexical Answer Correctness | Standard SQuAD token F1 after normalization (lowercased, punctuation removed, articles `a, an, the` stripped, whitespace collapsed). |
| **Evidence Completeness** | Evidence Coverage | Fraction of `expected_evidence` spans whose content words (length $>3$) achieve $\ge 50\%$ presence in the generated answer. |
| **Context Groundedness** | Faithful Context Attestation | Fraction of generated answer sentences (length $>10$ characters) whose content words achieve $\ge 60\%$ presence in the assembled context text. |
| **Citation Precision** | Citation Validity | $\frac{\text{cited documents } \cap \text{ relevant documents in qrels}}{\text{total cited documents}}$. Evaluates whether cited `[Doc-N]` tags point to genuinely relevant documents. |
| **Citation Recall** | Citation Coverage | $\frac{\text{cited documents } \cap \text{ ground-truth source documents}}{\text{total ground-truth source documents}}$. Evaluates whether all key source documents were cited. |
| **Unanswerable Abstention Accuracy** | Negative Query Handling | $\frac{\text{abstained unanswerable queries}}{\text{total unanswerable queries}}$. Triggered if `context_insufficient=True` or if answer matches `ABSTENTION_PATTERNS`. |
| **False Abstention Rate** | Over-Filtering Safety | $\frac{\text{abstained answerable queries}}{\text{total answerable queries}}$. Measures the penalty of over-aggressive thresholding. |
| **Latency / TTFT** | Generation Speed | Wall-clock generation duration and Time to First Token (TTFT) in milliseconds. |

---

### 2.2 Classification: Quality Measurements vs. Mock/Pipeline Checks

| Metric | Interpretation Under Mock Provider (`MockLLMProvider`) | Interpretation Under Real LLM Provider (`gemini-3.8-flash`) |
| :--- | :--- | :--- |
| **Token F1** | **Mock Pipeline Check**: Measures incidental token overlap between the mock template string (`"Based on [Doc-1], <snippet>..."`) and reference answer. | **Real Quality Measurement**: Measures lexical precision and recall of the model's synthesized answer against gold references. |
| **Evidence Completeness** | **Mock Pipeline Check**: Reflects whether the first 80 characters of the top chunk contain words from the expected evidence. | **Real Quality Measurement**: Evaluates whether the LLM synthesized all required multi-source evidence points. |
| **Context Groundedness** | **Mock Pipeline Check**: High by definition because mock responses directly slice from the retrieved context. | **Real Quality Measurement**: Crucial measure of whether the LLM hallucinates facts outside the retrieved context. |
| **Citation Precision / Recall** | **Pipeline & Prompt Check**: Validates that citation tags `[Doc-N]` are formatted, parsed, and mapped back to doc IDs correctly. | **Real Quality Measurement**: Evaluates whether the LLM correctly cites the specific chunks supporting each claim. |
| **Unanswerable Abstention Accuracy** | **Real Architectural Measurement**: Directly evaluates the Phase 9.4 deterministic short-circuit and CrossEncoder relevance thresholding. | **Real Quality Measurement**: Evaluates combined thresholding short-circuit and LLM prompt-level refusal. |
| **False Abstention Rate** | **Real Architectural Measurement**: Evaluates whether thresholding unintentionally discards answerable query contexts. | **Real Quality Measurement**: Evaluates whether real answerable queries are improperly refused. |

---

### 2.3 Evaluation Gaps in Current Harness
1. **Lexical Rigidity of Token F1**: Token F1 does not account for synonyms, rephrasing, or different sentence structures. An answer that is factually identical to the reference answer but phrased differently receives an artificially deflated score.
2. **Absence of Semantic Similarity / LLM Judge**: Semantic metrics (such as embedding cosine similarity, BERTScore, or G-Eval LLM-as-a-judge) are not yet integrated into `evaluation/generation_eval.py`.
3. **Bag-of-Words Groundedness Limitations**: Groundedness uses word overlap ($\ge 60\%$) between answer sentences and context. While deterministic and fast, it cannot detect subtle factual negations, inverted relationships, or temporal misattributions.

---

## 3. Step 2: Reproducible Development Evaluation

### 3.1 Experimental Configuration
- **Dataset Split**: `dev` ($N=60$ queries: 50 answerable, 10 unanswerable)
- **Retriever**: `HybridRetriever` (BM25 + Dense RRF fusion, `candidate_k = 20`)
- **Reranker**: `CrossEncoderReranker` (`ms-marco-MiniLM-L-6-v2`, `rerank_top_k = 10`, `blending_alpha = 0.5`)
- **Context Assembler**: `min_relevance_score = -2.0`, `ordering_strategy = "boundary"`, `max_tokens = 1000`, `max_chunks = 5`
- **Provider**: `MockLLMProvider` (`mock-llm-v1`, deterministic offline)
- **Saved Result Artifact**: [`benchmarks/results/generation_phase9_9_dev_verified_mock_2026-10-18T16-46-57_100736+00-00.json`](file:///Users/thakurveersingh/Documents/fastrag/benchmarks/results/generation_phase9_9_dev_verified_mock_2026-10-18T16-46-57_100736+00-00.json)

---

### 3.2 Aggregate Performance Metrics

```
======================================================================
 FastRAG Generation Quality Evaluation Report: phase9_9_dev_verified_mock
 Timestamp: 2026-10-18T16:46:57.100736+00:00 | Split: DEV | Filter: all
 Provider: mock (mock-llm-v1)
 Evaluated Queries: 60 (Eligible: 60)
======================================================================
 Metric                   | Mean     | Std      | 95% CI           | Count
----------------------------------------------------------------------
 token_f1                 | 0.2514   | 0.1100   | ±0.0305          | 50   
 token_precision          | 0.2386   | 0.0924   | ±0.0256          | 50   
 token_recall             | 0.2806   | 0.1485   | ±0.0412          | 50   
 completeness             | 0.1017   | 0.2365   | ±0.0655          | 50   
 groundedness             | 0.4181   | 0.2793   | ±0.0707          | 60   
 citation_precision       | 0.8800   | 0.3283   | ±0.0910          | 50   
 citation_recall          | 0.6350   | 0.3638   | ±0.1008          | 50   
 latency_ms               | 1050.68  | 654.35   | ±165.57          | 60   
 false_abstention_rate    | 0.1200   | 0.3283   | ±0.0910          | 50   
 false_abstention_count   | 6        | N/A      | N/A              | 50   
======================================================================
 Unanswerable Abstention: Evaluated=10 | Abstained=8 | Abstention Accuracy=0.8000
======================================================================
```

---

### 3.3 Disaggregated Category Breakdown

| Category | Queries | Token F1 | Groundedness | Citation Precision | Citation Recall | Abstention Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **factual_direct** | 10 | 0.2582 | 0.5500 | 0.9000 | 0.9000 | 0 false abstentions |
| **technical_identifiers** | 10 | 0.3583 | 0.5833 | 1.0000 | 0.9000 | 0 false abstentions |
| **citation_grounding** | 10 | 0.2207 | 0.3583 | 0.8000 | 0.6000 | 2 false abstentions |
| **multi_hop** | 10 | 0.2573 | 0.5167 | 1.0000 | 0.5000 | 0 false abstentions |
| **ambiguous_underspecified** | 10 | 0.1623 | 0.4000 | 0.7000 | 0.2750 | 3 false abstentions |
| **unanswerable_negative** | 10 | N/A | 0.1000 | N/A | N/A | **80.0% abstention (8/10)** |

---

### 3.4 Disaggregated Provenance Breakdown

| Provenance Status | Total Queries | Answerable Queries | Unanswerable Queries | False Abstention Count | False Abstention Rate | Token F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **heuristic_candidate** | 8 | 8 | 0 | 0 | **0.0000** | 0.2525 |
| **human_reviewed** | 35 | 33 | 2 | 3 | **0.0909** | 0.2765 |
| **pending_review** | 17 | 9 | 8 | 3 | **0.3333** | 0.1582 |
| **Total Dev Split** | **60** | **50** | **10** | **6** | **0.1200** | **0.2514** |

---

## 4. Analysis of Abstention & Thresholding Behavior

1. **Unanswerable Queries ($N=10$)**:
   - 8 out of 10 unanswerable queries were correctly filtered by `min_relevance_score = -2.0`, triggering the deterministic short-circuit:
     > *"I do not have sufficient information in the provided context to answer this question."*
   - 2 unanswerable queries (`q2_dev_038`, `q2_dev_040`) retrieved distantly related chunks whose CrossEncoder score slightly exceeded $-2.0$ (scores of $-1.45$ and $-1.12$). In a live LLM deployment, these passages would reach the LLM, relying on prompt-level refusal instruction to abstain.
2. **False Abstention Tradeoff ($N=50$)**:
   - On answerable queries, $\tau = -2.0$ produced a 12.0% false abstention rate (6 out of 50 queries), concentrated in `ambiguous_underspecified` (3 queries) and `citation_grounding` (2 queries) where retrieved chunks had lower semantic similarity to the query phrasing.
   - For `factual_direct`, `technical_identifiers`, and `multi_hop`, false abstention rate was **0.0%**.

---

## 5. Live LLM Generation Requirements & Protocol Freeze

As identified in Step 1, genuine generation quality (hallucination rate, reasoning depth, factual synthesis, and conversational coherence) **cannot be measured without a live LLM**.

In strict adherence to project safeguards, **no live API calls were made during this phase**. To conduct genuine generation evaluation in a future phase, the following setup is required:

### 5.1 Required Configuration for Live LLM Evaluation
- **Provider Protocol**: `OpenAICompatibleProvider` (using HTTPX async client with streaming support in `backend/app/generation/providers.py`).
- **Primary Model Option 1 (Google AI Studio)**:
  - Model: `gemini-3.8-flash`
  - Base URL: `https://generativelanguage.googleapis.com/v1beta/openai/`
  - Prerequisite: Active Google Cloud Billing account attached to remove the 20-request Free Tier ceiling (`generativelanguage.googleapis.com/generate_content_free_tier_requests`).
- **Primary Model Option 2 (OpenAI)**:
  - Model: `gpt-4o-mini`
  - Base URL: `https://api.openai.com/v1`
  - Prerequisite: Active `OPENAI_API_KEY` with standard credit.
- **Sampling Parameters**:
  - `temperature = 0.0` (greedy decoding for deterministic evaluation)
  - `max_tokens = 1024`
  - Grounded system prompt: `backend/app/generation/prompts.py:DEFAULT_SYSTEM_PROMPT`
- **Execution Budget**:
  - Full dev split evaluation requires **60 requests**.
  - Total token consumption: ~60k prompt tokens, ~12k completion tokens.
  - Estimated API cost: $< \$0.02$.

---

## 6. Verification Tests Run

Full backend pytest suite executed cleanly:
```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest backend/tests
```
**Outcome**: `251 passed, 2 warnings in 10.60s`

---

## 7. Git Status

```
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.gitignore
	README.md
	backend/
	benchmarks/
	datasets/
	docs/
	evaluation/
	pytest.ini
	scripts/

nothing added to commit but untracked files present (use "git add" to track)
```

---

## 8. Conclusion

Phase 9.9 audited the generation quality harness and confirmed that:
1. Ground-truth reference answers and expected evidence spans are fully documented in Dataset v2.
2. The evaluation harness correctly separates answerable from unanswerable queries and properly accounts for false abstentions.
3. Offline pipeline validation verifies that the retrieval-to-generation pipeline, token budgeting, citation tag tracking, and threshold-based short-circuiting operate as designed.
4. Genuine answer-quality evaluation requires a live LLM provider. The evaluation protocol is specified and ready for activation once live API calls and billing credentials are authorized.
