# FastRAG Phase 9.10: Live LLM Generation Evaluation Preparation Report

**Date**: October 18, 2026  
**Status**: Preparation Completed — Awaiting Approval for Live Execution  
**Branch**: `main`  
**Scope**: Preflight protocol and resource planning for live LLM generation evaluation on the development split; request volume, retry calculation, and authoritative pricing; error and rate-limit classification; and execution safeguards.

---

## 1. Executive Summary

Phase 9.10 establishes the operational and scientific protocol for executing a controlled live LLM generation evaluation of FastRAG. In Phase 9.9, the generation pipeline, token budgeting, and deterministic threshold abstention were verified offline using `MockLLMProvider`. 

This preparation report defines the exact parameters, request volume, costs, and failure safeguards required to transition from mock pipeline validation to genuine LLM answer-quality evaluation without unapproved spend or quota violations.

> [!IMPORTANT]
> **Strict Safeguard Enforcement**: In accordance with project instructions, **zero live API calls were made** during this phase. No keys were printed or logged. The held-out test split remains quarantined.

---

## 2. Verified Provider and Model Configuration

### 2.1 Primary Live Configuration (Google AI Studio Gemini)
- **Client Protocol**: `OpenAICompatibleProvider` (using asynchronous HTTPX client with connection pooling and timeouts).
- **Target Model Identifier**: `gemini-3.8-flash` (or `gemini-2.5-flash`).
- **Endpoint URL**: `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`
- **Authentication**: `Bearer <API_KEY>` supplied via `FASTRAG_LLM_API_KEY` or `LLM_API_KEY` (configured in `.env`).
- **Sampling Temperature**: `0.0` (greedy decoding for deterministic, reproducible evaluation).
- **Maximum Completion Tokens**: `1024` tokens.
- **Request Timeout**: `30.0` seconds.
- **Pacing Delay**: `4.0` seconds inter-query pacing delay to stay safely under RPM ceilings.

### 2.2 Alternative Live Configuration (OpenAI Direct)
- **Target Model Identifier**: `gpt-4o-mini`
- **Endpoint URL**: `https://api.openai.com/v1/chat/completions`
- **Authentication**: `Bearer <API_KEY>` supplied via `OPENAI_API_KEY`.
- **Sampling Parameters**: `temperature = 0.0`, `max_tokens = 1024`.

---

## 3. Exact Request-Count & Retry Calculation

### 3.1 Target Query Set
- **Split**: Development split (`datasets/v2/queries_dev.jsonl`).
- **Total Queries**: **60**
  - **Answerable Queries**: 50 (factual, multi-hop, technical, citation, ambiguous)
  - **Unanswerable Queries**: 10 (negative unanswerable questions)

### 3.2 Request Volume by Execution Mode

#### Option A: Thresholded & Short-Circuited Pipeline (Recommended Production Setting)
- Context Assembler setting: `min_relevance_score = -2.0`, `ordering_strategy = "boundary"`.
- Under this setting, queries whose retrieved chunks all score below $-2.0$ are marked `context_insufficient = True`. The pipeline immediately short-circuits in [`backend/app/generation/generator.py`](file:///Users/thakurveersingh/Documents/fastrag/backend/app/generation/generator.py#L125) and returns the canonical abstention message without calling the provider.
- Empirical dev split reduction:
  - **8 unanswerable queries** short-circuit $\to$ **0 API requests**.
  - **2 unanswerable queries** exceed threshold $\to$ **2 API requests** (testing prompt refusal).
  - **6 answerable queries** below threshold (false abstentions) $\to$ **0 API requests**.
  - **44 answerable queries** exceed threshold $\to$ **44 API requests**.
- **Nominal Primary Requests**: **46 requests**.

#### Option B: Unconstrained Baseline Pipeline
- Context Assembler setting: `min_relevance_score = None`.
- All candidate chunks pass into the prompt regardless of score.
- **Nominal Primary Requests**: exactly **60 requests**.

### 3.3 Retry Policy & Worst-Case Request Ceiling
- **Configured Maximum Retries**: `max_retries = 3` (exponential backoff factor `2.0x`, initial delay `1.0s`, random jitter).
- **Fail-Fast Exemptions (0 Retries)**:
  - `QUOTA_EXHAUSTED` (HTTP 429 quota exceeded): **0 retries** (immediate abort).
  - `AUTHENTICATION_ERROR` (HTTP 401/403): **0 retries** (immediate abort).
- **Eligible for Retry (up to 3 attempts)**:
  - Transient RPM/TPM rate limits (HTTP 429).
  - Upstream provider server errors (HTTP 500, 502, 503, 504).
  - Network transport errors and timeouts.
- **Worst-Case Request Ceiling**: $60 \times 4 = 240$ requests (only reached if all queries fail 3 times before succeeding).
- **Expected Actual Requests**: **46 to 60 requests** under normal network conditions.

---

## 4. Authoritative Pricing & Realistic Cost Estimate

Authoritative pricing was verified from official provider documentation (Google AI Studio and OpenAI):

### 4.1 Authoritative Unit Rates

| Provider / Model | Authoritative Source | Input Tokens (per 1M) | Output Tokens (per 1M) |
| :--- | :--- | :---: | :---: |
| **Google Gemini 3.8 Flash** | Google AI Studio Documentation (2026 standard rates) | **$0.75** | **$3.75** |
| **Google Gemini 2.5 Flash** | Google AI Studio Documentation | **$0.30** | **$2.50** |
| **OpenAI GPT-4o-mini** | OpenAI API Pricing Documentation | **$0.15** | **$0.60** |

### 4.2 Token Volume Projection for Full Dev Evaluation ($N=60$ Queries)
- **Context Budget Limit**: 1,000 tokens maximum (`context_max_tokens = 1000`).
- **Average Retrieved Context**: ~550 tokens.
- **System Prompt & Query Overhead**: ~180 tokens.
- **Average Prompt Size per Request**: ~730 tokens.
- **Average Completion Size per Request**: ~200 tokens.

| Scenario | Mode | Prompt Tokens | Completion Tokens | Total Tokens |
| :--- | :--- | :---: | :---: | :---: |
| **Unconstrained ($N=60$)** | Baseline | ~43,800 tokens (0.0438M) | ~12,000 tokens (0.0120M) | ~55,800 tokens |
| **Short-Circuited ($N=46$)** | Production | ~33,580 tokens (0.0336M) | ~9,200 tokens (0.0092M) | ~42,780 tokens |

### 4.3 Total Cost Calculation

| Model Option | Evaluation Mode | Input Cost | Output Cost | Total Estimated Cost |
| :--- | :--- | :---: | :---: | :---: |
| **Gemini 3.8 Flash** | Unconstrained ($N=60$) | $0.0438 \times \$0.75 = \$0.0329$ | $0.0120 \times \$3.75 = \$0.0450$ | **$0.078** (~7.8¢) |
| **Gemini 3.8 Flash** | Short-Circuited ($N=46$) | $0.0336 \times \$0.75 = \$0.0252$ | $0.0092 \times \$3.75 = \$0.0345$ | **$0.060** (~6.0¢) |
| **GPT-4o-mini** | Unconstrained ($N=60$) | $0.0438 \times \$0.15 = \$0.0066$ | $0.0120 \times \$0.60 = \$0.0072$ | **$0.014** (~1.4¢) |
| **GPT-4o-mini** | Short-Circuited ($N=46$) | $0.0336 \times \$0.15 = \$0.0050$ | $0.0092 \times \$0.60 = \$0.0055$ | **$0.011** (~1.1¢) |

> [!NOTE]
> Even under a worst-case scenario with multiple transient retries, the total cost for the entire development evaluation remains **under \$0.25 USD**.

---

## 5. Error Handling, Rate Limits, and Malformed Responses

The evaluation pipeline includes defensive mechanisms implemented in [`scripts/run_external_model_comparison.py`](file:///Users/thakurveersingh/Documents/fastrag/scripts/run_external_model_comparison.py) and [`backend/app/generation/providers.py`](file:///Users/thakurveersingh/Documents/fastrag/backend/app/generation/providers.py):

1. **Hard Quota Exhaustion (`QUOTA_EXHAUSTED`)**:
   - Detects `RESOURCE_EXHAUSTED`, `free_tier_requests`, and HTTP 429 quota error strings.
   - Raises `ProviderQuotaExhaustedError` and aborts immediately on Attempt 1.
   - Prevents wasting backoff delays or hammering exhausted endpoints.
2. **Transient Rate Limits (`RATE_LIMIT_RPM`, `RATE_LIMIT_TPM`)**:
   - Detects temporary rate throttling and extracts `retry-after` header if present.
   - Applies exponential backoff with jitter and retries up to 3 times.
3. **Authentication Errors (`AUTHENTICATION_ERROR`)**:
   - Detects HTTP 401/403 or invalid key responses.
   - Aborts immediately on Attempt 1; redacts all authorization headers and credentials.
4. **Malformed Responses & Server Errors**:
   - Validates JSON payload schema (`choices[0].message.content`).
   - If response payload is incomplete or empty, records `raw_error` in the query record and safely records 0.0 scores rather than crashing the batch run.
5. **Zero Silent Fallback**:
   - Live runs never silently substitute mock outputs when a provider call fails. Errors are recorded transparently in benchmark telemetry.

---

## 6. Proposed Command for Dev-Only Live Evaluation

When authorized, the live generation evaluation will be executed using the existing CLI harness with explicit experiment naming:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 scripts/run_generation_eval.py \
  --split dev \
  --provenance all \
  --retriever hybrid-rerank \
  --provider gemini \
  --model gemini-3.8-flash \
  --candidate-k 20 \
  --rerank-top-k 10 \
  --min-relevance-score -2.0 \
  --ordering-strategy boundary \
  --blending-alpha 0.5 \
  --temperature 0.0 \
  --experiment-id live_gemini_3_8_flash_dev_all_alpha0.5_thresh-2.0
```

### Expected Output Artifact:
- **File Path**: `benchmarks/results/generation_live_gemini_3_8_flash_dev_all_alpha0.5_thresh-2.0_<TIMESTAMP>.json`
- **Output Properties**:
  - `split`: `"dev"`
  - `provider_type`: `"openai_compatible"`
  - `model_identifier`: `"gemini-3.8-flash"`
  - `eligible_query_count`: `60`
  - Complete `summary_metrics`, `category_metrics`, `provenance_metrics`, and `unanswerable_metrics`.

---

## 7. Blockers & Decisions Requiring Approval

The following items are identified for user decision prior to live execution:

1. **Google AI Studio Quota vs. Billing**:
   - *Current Status*: The Google AI Studio Free Tier ceiling (`generativelanguage.googleapis.com/generate_content_free_tier_requests`, limit 20 requests) was reached during Phase 9.3 testing.
   - *Requirement*: To execute the 46–60 requests needed for the full dev split, a Google Cloud project with Pay-As-You-Go billing attached is required (or a fresh API key with available quota).
2. **Model Choice Confirmation**:
   - Confirm whether to use **`gemini-3.8-flash`** (configured in `.env`) or alternative **`gpt-4o-mini`** (requires providing `OPENAI_API_KEY`).
3. **Sandbox Network Approval**:
   - Standard sandbox mode blocks outbound socket connections (`[Errno 8] nodename nor servname provided, or not known`).
   - Live API calls will require executing the single runner command with user-approved network access (`BypassSandbox: true`).
4. **Execution Scope**:
   - Confirm whether to run the full dev split (60 queries, ~46–60 requests, ~$0.08) or a fast 5-query validation smoke test first.

---

## 8. Conclusion

All prerequisites, configuration schemas, token budgets, pricing sources, and error-handling paths for live generation evaluation have been verified and documented. Execution has stopped per protocol. Live API calls await user authorization.
