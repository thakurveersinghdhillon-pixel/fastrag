# FastRAG Dataset v2: Annotation Guidelines

## 1. Overview and Purpose
These guidelines define the standards for evaluating and labeling questions, evidence, reference answers, and retrieval relevance for **FastRAG Dataset v2**.
The goal is to establish a rigorous, reproducible research benchmark suitable for comparing retrieval architectures (dense, BM25, hybrid, reranked) and LLM generation quality (groundedness, citation accuracy, unanswerable detection).

---

## 2. Relevance Grading Criteria (IR Phase)

Every query-document pair in `qrels.tsv` is assigned an integer grade in `{0, 1, 2}` following standard TREC conventions:

### Grade 2: Highly Relevant (Direct Evidence)
- **Definition**: The document directly contains the core factual information required to answer the query completely or forms an indispensable premise in a multi-hop reasoning chain.
- **Examples**:
  - Query: *"What is the formula for Reciprocal Rank Fusion?"*
  - Document `doc_014` (*"RRF Mathematical Principles"*): **Grade 2** (directly states `sum(1 / (k + rank))`).
  - Query: *"What are the parameters for HNSW graphs?"*
  - Document `doc_007` (*"HNSW Graphs"*): **Grade 2** (defines `M` and `efConstruction`).

### Grade 1: Partially Relevant (Contextual / Background Evidence)
- **Definition**: The document mentions related concepts, provides helpful contextual background, discusses tuning of the parameters, or covers a sister architecture, but does not directly or fully answer the question in isolation.
- **Examples**:
  - Query: *"What is the formula for Reciprocal Rank Fusion?"*
  - Document `doc_015` (*"RRF Smoothing Constant k Tuning"*): **Grade 1** (explains tuning of $k$ but does not define the full formula).
  - Document `doc_016` (*"Deduplication in Hybrid Pipelines"*): **Grade 1** (discusses RRF deduplication benefits).

### Grade 0: Not Relevant (Irrelevant / Distractor / Unanswerable)
- **Definition**: The document does not contain relevant information, is an explicit distractor (e.g. database B-Trees, CPU hardware cache lines, e-commerce search), or the query is fundamentally unanswerable from the corpus.
- **Rule for `unanswerable_negative` queries**: All documents in the corpus receive **Grade 0**.

---

## 3. Answer Correctness & Evidence Extraction (Generation Phase)

### 3.1 Reference Answer Formulation
1. **Factual Direct**: The answer must state the exact fact directly supported by the source document. No extrapolations or unstated assumptions.
2. **Multi-Hop**: The answer must synthesize evidence from all referenced `source_doc_ids`. If an answer omits the connection between the documents, it is incomplete.
3. **Unanswerable / Negative**: The reference answer must explicitly be marked `null` (or a standardized abstention message such as *"The provided corpus does not contain sufficient evidence to answer this question"*). The model must refrain from fabricating speculative details.
4. **Ambiguous / Underspecified**: The reference answer must delineate the multiple interpretations supported by the corpus (e.g., distinguishing Level 1 RAG response cache from Level 2 vector cache and hardware CPU cache).

### 3.2 Expected Evidence Spans
- Each answerable query record contains `expected_evidence`, a list of exact or near-exact textual substrings from the source documents.
- Verifiers must confirm that every listed evidence span exists verbatim in the corresponding document `text`.

---

## 4. Groundedness and Faithfulness Criteria

A generated answer is evaluated along the **RAG Triad**:

1. **Context Relevance**: Did the retriever fetch passages that address the user's information need without excessive noise?
2. **Faithfulness**:
   - Every claim in the generated answer must be logically entailed by the retrieved context.
   - Any claim not found in the context—even if factually true in the general world—is penalized as an ungrounded assertion.
3. **Answer Relevance**: Did the model directly address the prompt without tangential deviation?

---

## 5. Citation Correctness Guidelines

For queries in the `citation_grounding` category (and general generation auditing):
1. **Placement**: Citations must be placed immediately following the assertion they support:
   ```text
   Recursive character chunking uses a hierarchy of double newlines, single newlines, and spaces [doc_001].
   ```
2. **Precision**: The cited document must actually contain the asserted fact. Citing a partially relevant background document (`doc_004`) for a recursive chunking claim (`doc_001`) is penalized as a hallucinated citation.
3. **Completeness**: If a sentence draws upon two distinct sources, both must be cited (e.g., `[doc_005][doc_017]`).

---

## 6. Annotation Provenance Protocol

All query records must specify their review provenance in `annotation_provenance`:
- **`status`**: Must be one of `human_reviewed`, `heuristic_candidate`, or `pending_review`.
- **`author`**: Identifying team or annotator ID.
- **`review_date`**: ISO 8601 date of review.
- **`method`**: Description of creation method (`manual_expert_curation`, `multi_topic_pairing`, `manual_negative_design`, etc.).

> [!IMPORTANT]
> Annotators must **never** mark an automated or heuristic query as `human_reviewed` without manual inspection of both the query text and all corresponding source evidence spans.
