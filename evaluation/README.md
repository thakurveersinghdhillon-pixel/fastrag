# FastRAG Evaluation Framework

This package provides the evaluation metrics, dataset validation, experiment metadata tracking, and evaluation runners for measuring retrieval quality and system reproducibility in FastRAG.

---

## 1. Package Structure

```
evaluation/
├── __init__.py               # Evaluation package entry point
├── dataset_validation.py     # Schema, referential integrity, and SHA-256 validation
├── metrics.py                # Standalone IR metrics: Recall@K, MRR@K, nDCG@K
├── runner.py                 # EvaluationRunner orchestrating batch metric calculation
├── experiment.py             # ExperimentMetadata linking runs to dataset manifest hash
└── fixtures.py               # Deterministic synthetic test fixtures (perfect, partial, inverse, empty)
```

---

## 2. Implemented Information Retrieval Metrics

All metrics operate strictly on ranked lists of document identifiers and ground-truth relevance mappings (`qrels`):

### A. Recall@K
Calculates the fraction of total relevant documents captured in the top-$K$ candidates:
$$\text{Recall@}K = \frac{|\{d \in \text{retrieved}_{1..K}\} \cap \{d : \text{rel}(d) > 0\}|}{|\{d : \text{rel}(d) > 0\}|}$$
- **Scale**: Binary relevance ($\text{rel} > 0$).
- **Edge cases**: Returns `0.0` if no relevant documents exist in ground truth or if retrieved list is empty.

### B. Reciprocal Rank / MRR@K
Measures how early the first relevant hit appears in the ranking:
$$\text{RR@}K = \frac{1}{\text{rank}^*} \quad (\text{if } \text{rank}^* \le K, \text{ else } 0)$$
where $\text{rank}^*$ is the 1-based index of the first relevant document ($\text{rel} > 0$).
- **Mean Reciprocal Rank (MRR)**: Evaluated as the arithmetic mean of $\text{RR@}K$ across all queries in the evaluation set.

### C. Normalized Discounted Cumulative Gain (nDCG@K)
Evaluates graded relevance ordering ($0 = \text{not relevant}$, $1 = \text{partially relevant}$, $2 = \text{highly relevant}$):
$$\text{DCG@}K = \sum_{i=1}^{\min(K, N)} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$
$$\text{nDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$
where $\text{IDCG@}K$ is the ideal DCG obtained by sorting all positive judgments in descending order. Returns `0.0` if ideal gain is zero.

---

## 3. Evaluation Runner & CLI

### Programmatic Usage

```python
from evaluation.runner import EvaluationRunner, load_qrels

qrels = load_qrels("datasets/qrels.tsv")
rankings = {"q_001": ["doc_001", "doc_004"], "q_002": ["doc_009"]}

runner = EvaluationRunner(qrels=qrels, metrics=["recall@5", "mrr@10", "ndcg@10"])
report = runner.run(rankings=rankings, experiment_id="my_experiment")

print(report.summary())
```

### CLI Execution

```bash
# Evaluate using synthetic verification fixtures:
python3 scripts/run_eval.py --fixture perfect --no-save
python3 scripts/run_eval.py --fixture partial --no-save

# Evaluate against a custom rankings JSON file:
python3 scripts/run_eval.py --rankings path/to/rankings.json --experiment-id my_run
```
