# RL RAG SLO Controller

This repository contains an experimental framework for **RL-controlled Retrieval-Augmented Generation (RAG)** under explicit **Service Level Objectives (SLOs)**.

The core idea: treat RAG configuration (retrieval depth, model size, refusal behavior, etc.) as a **discrete action space**, and learn a **contextual bandit policy** that chooses the best action per query to trade off:

- Answer quality  
- Token / cost budget  
- Hallucination penalties  
- Refusal behavior  

The initial setup uses **SQuAD 2.0** and is intended as a foundation for an **RL + RAG** research paper.

---

## Features

- **Contextual bandit controller** over discrete RAG actions (k, model size, answer mode).
- **Pluggable retriever**:
  - BM25 via `rank_bm25` (if installed),
  - TF-IDF cosine similarity fallback via `scikit-learn`.
- **Pluggable LLM backend**:
  - `DummyLLMClient` (no external API calls) — for fast pipeline testing.
  - `OpenAILLMClient` (optional) — for real generations via OpenAI API.
- **SLO-aware reward** combining:
  - answer accuracy,
  - token/cost penalty,
  - hallucination penalty,
  - refusal behavior (correct vs wrong refusals).
- **Dataset loaders**:
  - SQuAD 2.0 (answerable + unanswerable QA),
  - HotpotQA (for later multi-hop / long-context experiments).
- **Offline RL workflow**:
  - replay buffer generation from logged (s, a, r),
  - training a contextual bandit policy,
  - evaluating against a fixed RAG baseline.

---

## Project Structure

    rl_rag_slo/
      README.md
      rl_rag_slo/
        controller/
          actions.py           # discrete RAG action definitions
          bandit_trainer.py    # contextual bandit training
          policy_network.py    # PyTorch MLP policy
          slo_profiles.py      # SLO weight presets
          state_encoder.py     # (query, SLO, meta) -> state vector
        datasets/
          squad2_loader.py     # SQuAD 2.0 loader + corpus builder
          hotpot_loader.py     # HotpotQA loader + corpus builder
        env/
          rag_env.py           # single-step RAG environment with reward
        llm_backend/
          llm_client.py        # BaseLLMClient, DummyLLMClient, OpenAILLMClient
          answer_scorer.py     # EM-style metrics + hallucination/refusal flags
          refusal_detector.py  # heuristic refusal detector
          embedding_client.py  # BaseEmbedder, DeterministicHashEmbedder, OpenAIEmbedder
        retrievers/
          bm25_retriever.py    # BM25 / TF-IDF-based retriever
        scripts/
          generate_logs.py     # build offline replay buffer (SQuAD)
          train_policy.py      # train contextual bandit policy
          eval_policy.py       # evaluate policy vs baseline
          smoke_test.py        # quick end-to-end sanity check

---

## Installation

This is a standard Python project (tested with Python 3.10+).

1. Create and activate a virtual environment (recommended).
2. Install dependencies, for example:

    pip install numpy torch scikit-learn rank-bm25 openai tqdm

Notes:

- `rank-bm25` is optional (falls back to TF-IDF if missing).
- `openai` is only needed if you use `OpenAILLMClient` or `OpenAIEmbedder`.
- The **default** scripts use `DummyLLMClient` and a deterministic hash-based embedder, so they will run without any external APIs.

---

## Reproducibility

Environment setup:

- Recommended Python version: 3.10+
- Install dependencies: `pip install -r requirements.txt` (if present) or `pip install -e .`
- If you use the OpenAI backend, set `OPENAI_API_KEY` in your environment

Dataset note:

- You must download SQuAD v2.0 JSON files and pass the train/dev paths explicitly.

Main pipeline (copy/paste):

```bash
python -m rl_rag_slo.scripts.precompute_replay_multi_slo --squad_path <PATH_TO_TRAIN_V2_JSON> --output_path replay_squad2_multi.npz --num_examples 1000 --slo_profiles "quality_first,cheap"

python -m rl_rag_slo.scripts.train_policy --replay_path replay_squad2_multi.npz --output_path policy_squad2_multi_argmax.pt --epochs 10 --batch_size 128 --device cpu --objective argmax_ce

mkdir -p results
python -m rl_rag_slo.scripts.eval_policy --model_path policy_squad2_multi_argmax.pt --squad_path <PATH_TO_DEV_V2_JSON> --num_examples 200 --device cpu --slo_profile quality_first --output_json results/quality_first_argmax.json
python -m rl_rag_slo.scripts.eval_policy --model_path policy_squad2_multi_argmax.pt --squad_path <PATH_TO_DEV_V2_JSON> --num_examples 200 --device cpu --slo_profile cheap --output_json results/cheap_argmax.json

python -m rl_rag_slo.scripts.make_figures --inputs results/quality_first_argmax.json results/cheap_argmax.json --out_dir results
```

Expected outputs:

- `results/summary.csv`
- `results/action_distribution.png`
- `results/cost_vs_accuracy.png`
- `results/reward_vs_baseline.png`

Note: Runs can be slow due to LLM calls.

---

## Datasets

### SQuAD 2.0

- Download the official SQuAD v2.0 JSON file (train or dev) from the original source.
- Keep the file locally (e.g., `/data/squad2_train.json`).
- Pass the path to scripts via `--squad_path`.

The loader in `datasets/squad2_loader.py` expects the standard v2.0 structure and produces:

- a flat list of `QAExample` objects,
- a deduplicated context corpus for retrieval.

### HotpotQA

- The loader in `datasets/hotpot_loader.py` supports the typical HotpotQA JSON format (train/dev).
- It is not used by the initial experiments but is available for:
  - multi-hop reasoning,
  - long-context / multi-document retrieval,
  - future extensions of the RL + RAG experiments.

Both datasets are assumed to be **provided as local JSON files**. The repo does not download them automatically.

---

## Quickstart: Smoke Test

Before doing any RL experiments, run a **smoke test** with the dummy LLM client to ensure the wiring is correct.

1. Download SQuAD v2.0 JSON and note the path (e.g. `/path/to/squad2.json`).

2. Run:

    python -m rl_rag_slo.scripts.smoke_test \
        --squad_path /path/to/squad2.json \
        --num_examples 20

This will:

- load SQuAD 2.0 examples,
- build a BM25/TF-IDF index over contexts,
- run a simple RAG pipeline with `DummyLLMClient`,
- print per-example:
  - question,
  - chosen action id,
  - reward and token cost,
  - accuracy / hallucination / refusal flags.

If this completes without errors, the basic plumbing is working.

---

## Offline RL Workflow

The main RL loop uses **offline contextual bandit training**:

1. Generate replay data  
2. Train a policy  
3. Evaluate vs a fixed RAG baseline  

### Step 1 – Generate Replay Data

    python -m rl_rag_slo.scripts.generate_logs \
        --squad_path /path/to/squad2.json \
        --output_path replay_squad2_small.npz \
        --num_examples 5000

This script:

- loads SQuAD 2.0 QAs,
- builds a retriever and RAG environment,
- samples actions uniformly from the discrete action set in `controller/actions.py`,
- runs one RAG step per question,
- logs:
  - state vector (encoded query + SLO),
  - chosen action id,
  - scalar reward,
- saves everything as `states`, `actions`, `rewards` in a `.npz` file.

### Step 2 – Train the Policy

    python -m rl_rag_slo.scripts.train_policy \
        --replay_path replay_squad2_small.npz \
        --output_path policy_squad2.pt \
        --epochs 10 \
        --batch_size 128 \
        --device cpu

This script:

- loads the replay buffer,
- builds a `PolicyNetwork` (MLP) and `BanditTrainer`,
- trains using a simple REINFORCE-style objective on offline data,
- saves the model weights to `policy_squad2.pt`.

### Step 3 – Evaluate vs Baseline

    python -m rl_rag_slo.scripts.eval_policy \
        --model_path policy_squad2.pt \
        --squad_path /path/to/squad2.json \
        --num_examples 1000 \
        --device cpu

This script:

- evaluates a **fixed baseline** RAG configuration (e.g., k=5, base model, guarded),
- evaluates the **learned policy** on the same questions,
- compares:

  - `avg_accuracy`
  - `avg_cost_tokens`
  - `hallucination_rate`
  - `refusal_rate`

The goal is to see whether the learned policy finds better quality/cost/hallucination/refusal tradeoffs than the fixed baseline.

---

## Multi-SLO Replay (One Policy, Multiple SLOs)

Use the multi-SLO replay script to train a single SLO-conditioned policy, then evaluate it under different SLOs.

Precompute replay with both SLOs:

    python -m rl_rag_slo.scripts.precompute_replay_multi_slo `
      --squad_path "C:\Users\vanda\OneDrive\Desktop\RL-RAG\rl_rag_slo\datasets\SQuAD\train-v2.0.json" `
      --output_path "replay_squad2_multi.npz" `
      --num_examples 1000 `
      --slo_profiles "quality_first,cheap"

Train one policy:

    python -m rl_rag_slo.scripts.train_policy `
      --replay_path "replay_squad2_multi.npz" `
      --output_path "policy_squad2_multi.pt" `
      --epochs 10 `
      --batch_size 128 `
      --device cpu

Evaluate the same policy under different SLOs:

    # Quality-first evaluation
    python -m rl_rag_slo.scripts.eval_policy `
      --model_path "policy_squad2_multi.pt" `
      --squad_path "C:\Users\vanda\OneDrive\Desktop\RL-RAG\rl_rag_slo\datasets\SQuAD\dev-v2.0.json" `
      --num_examples 200 `
      --device cpu `
      --slo_profile quality_first

    # Cheap evaluation
    python -m rl_rag_slo.scripts.eval_policy `
      --model_path "policy_squad2_multi.pt" `
      --squad_path "C:\Users\vanda\OneDrive\Desktop\RL-RAG\rl_rag_slo\datasets\SQuAD\dev-v2.0.json" `
      --num_examples 200 `
      --device cpu `
      --slo_profile cheap

Success looks like:

- the action distribution should differ across SLOs
- reward should be higher under the matching SLO than naive fixed baselines

---

## LLM and Embedding Backends

By default, the scripts are configured to avoid external dependencies:

- **LLM**: `DummyLLMClient` — concatenates retrieved texts and returns a mock answer.
- **Embeddings**: deterministic hash-based vectors — stable, but not semantically meaningful.

For real experiments you can switch to **OpenAI-backed** components:

- `OpenAILLMClient` in `llm_backend/llm_client.py`
- `OpenAIEmbedder` in `llm_backend/embedding_client.py`

You must set:

    export OPENAI_API_KEY="sk-..."

before using these classes.

Be aware that using real models introduces **latency** and **cost**, so start small and scale up carefully.

---

## Paper/Blog Results Reproduction (Fast)

Run evaluation with JSON output for both SLOs and both policies, then generate plots.

Windows example (PowerShell):

    python -m rl_rag_slo.scripts.eval_policy `
      --model_path "C:\Users\vanda\OneDrive\Desktop\RL-RAG\policy_squad2_multi_argmax.pt" `
      --squad_path "C:\Users\vanda\OneDrive\Desktop\RL-RAG\rl_rag_slo\datasets\SQuAD\dev-v2.0.json" `
      --num_examples 200 `
      --device cpu `
      --slo_profile quality_first `
      --output_json "C:\Users\vanda\OneDrive\Desktop\RL-RAG\results\eval_argmax_quality_first.json"

    python -m rl_rag_slo.scripts.eval_policy `
      --model_path "C:\Users\vanda\OneDrive\Desktop\RL-RAG\policy_squad2_multi_argmax.pt" `
      --squad_path "C:\Users\vanda\OneDrive\Desktop\RL-RAG\rl_rag_slo\datasets\SQuAD\dev-v2.0.json" `
      --num_examples 200 `
      --device cpu `
      --slo_profile cheap `
      --output_json "C:\Users\vanda\OneDrive\Desktop\RL-RAG\results\eval_argmax_cheap.json"

    python -m rl_rag_slo.scripts.eval_policy `
      --model_path "C:\Users\vanda\OneDrive\Desktop\RL-RAG\policy_squad2_multi_argmax_wt.pt" `
      --squad_path "C:\Users\vanda\OneDrive\Desktop\RL-RAG\rl_rag_slo\datasets\SQuAD\dev-v2.0.json" `
      --num_examples 200 `
      --device cpu `
      --slo_profile quality_first `
      --output_json "C:\Users\vanda\OneDrive\Desktop\RL-RAG\results\eval_argmax_wt_quality_first.json"

    python -m rl_rag_slo.scripts.eval_policy `
      --model_path "C:\Users\vanda\OneDrive\Desktop\RL-RAG\policy_squad2_multi_argmax_wt.pt" `
      --squad_path "C:\Users\vanda\OneDrive\Desktop\RL-RAG\rl_rag_slo\datasets\SQuAD\dev-v2.0.json" `
      --num_examples 200 `
      --device cpu `
      --slo_profile cheap `
      --output_json "C:\Users\vanda\OneDrive\Desktop\RL-RAG\results\eval_argmax_wt_cheap.json"

    python -m rl_rag_slo.scripts.make_figures `
      --inputs "C:\Users\vanda\OneDrive\Desktop\RL-RAG\results\eval_argmax_quality_first.json" `
               "C:\Users\vanda\OneDrive\Desktop\RL-RAG\results\eval_argmax_cheap.json" `
               "C:\Users\vanda\OneDrive\Desktop\RL-RAG\results\eval_argmax_wt_quality_first.json" `
               "C:\Users\vanda\OneDrive\Desktop\RL-RAG\results\eval_argmax_wt_cheap.json" `
      --out_dir "C:\Users\vanda\OneDrive\Desktop\RL-RAG\results"

Generic paths:

    python -m rl_rag_slo.scripts.eval_policy \
      --model_path /path/to/policy_squad2_multi_argmax.pt \
      --squad_path /path/to/dev-v2.0.json \
      --num_examples 200 \
      --device cpu \
      --slo_profile quality_first \
      --output_json /path/to/results/eval_argmax_quality_first.json

    python -m rl_rag_slo.scripts.eval_policy \
      --model_path /path/to/policy_squad2_multi_argmax.pt \
      --squad_path /path/to/dev-v2.0.json \
      --num_examples 200 \
      --device cpu \
      --slo_profile cheap \
      --output_json /path/to/results/eval_argmax_cheap.json

    python -m rl_rag_slo.scripts.eval_policy \
      --model_path /path/to/policy_squad2_multi_argmax_wt.pt \
      --squad_path /path/to/dev-v2.0.json \
      --num_examples 200 \
      --device cpu \
      --slo_profile quality_first \
      --output_json /path/to/results/eval_argmax_wt_quality_first.json

    python -m rl_rag_slo.scripts.eval_policy \
      --model_path /path/to/policy_squad2_multi_argmax_wt.pt \
      --squad_path /path/to/dev-v2.0.json \
      --num_examples 200 \
      --device cpu \
      --slo_profile cheap \
      --output_json /path/to/results/eval_argmax_wt_cheap.json

    python -m rl_rag_slo.scripts.make_figures \
      --inputs /path/to/results/eval_argmax_quality_first.json \
               /path/to/results/eval_argmax_cheap.json \
               /path/to/results/eval_argmax_wt_quality_first.json \
               /path/to/results/eval_argmax_wt_cheap.json \
      --out_dir /path/to/results

---

## Next Steps / TODO

Planned extensions (useful for a paper roadmap):

- Multi-dataset / multi-domain experiments:
  - Add HotpotQA for multi-hop and longer reasoning chains.
- **SLO-conditioning**:
  - Include different SLO profiles as part of the state and train a single policy that adapts to SLO changes at query time.
- Richer action space:
  - Allow query reformulation, second-stage retrieval, or “ask again” actions.
- Domain-specific RAG:
  - Plug in clinical or other specialized corpora (e.g., MIMIC-like QA) for more realistic evaluations.
- Better reward shaping:
  - Move beyond exact-match to include more robust QA metrics and human preference-style feedback.

This repo is intentionally minimal and opinionated so it can serve as a clean **starting point** for RL + RAG research and a paper prototype.
