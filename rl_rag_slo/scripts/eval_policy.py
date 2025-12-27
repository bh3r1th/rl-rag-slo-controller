import argparse
import datetime
import json
import time
from typing import Dict

import numpy as np
import torch

from rl_rag_slo.datasets.squad2_loader import load_squad2_qa, build_squad2_corpus
from rl_rag_slo.retrievers.bm25_retriever import BM25Retriever
from rl_rag_slo.llm_backend.llm_client import DummyLLMClient, OpenAILLMClient
from rl_rag_slo.controller.state_encoder import StateEncoder
from rl_rag_slo.slo.slo_config import get_slo_vector, weights_for_profile
from rl_rag_slo.controller.feature_utils import (
    compute_question_features,
    compute_bm25_features,
)
from rl_rag_slo.env.rag_env import RagEnvironment
from rl_rag_slo.controller.bandit_trainer import BanditTrainer
from rl_rag_slo.controller.actions import ACTIONS
from rl_rag_slo.llm_backend.answer_scorer import compute_qa_score
from rl_rag_slo.metrics.retrieval_hit import (
    compute_retrieval_hit,
    compute_retrieval_hit_rate,
)


def deterministic_embed(question: str, dim: int = 128) -> np.ndarray:
    """
    Create a deterministic pseudo-random embedding for the question based on its hash.

    This version clamps the seed into the 32-bit range accepted by numpy.RandomState.
    """
    import hashlib

    h = hashlib.sha256(question.encode("utf-8")).digest()
    # Take 8 bytes, interpret as integer, then clamp into [0, 2**32 - 1]
    raw_seed = int.from_bytes(h[:8], "little", signed=False)
    seed = raw_seed % (2**32 - 1)
    rng = np.random.RandomState(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(dim,)).astype(np.float32)


def evaluate_policy(
    examples,
    env: RagEnvironment,
    state_encoder: StateEncoder,
    slo_vec: np.ndarray,
    policy_trainer: BanditTrainer,
    device: str,
    max_refusal_frac: float | None,
) -> Dict[str, float]:
    """
    Evaluate the learned policy on a list of QA examples.

    Returns a tuple:
      - metrics: dict with aggregated metrics
      - action_counts: dict mapping action_id -> count
    """
    import collections

    total_acc = 0.0
    total_cost = 0.0
    total_halluc = 0
    total_refusal = 0
    total_reward = 0.0
    n = 0

    action_counts = collections.Counter()
    answerable_flags = []
    retrieval_hits = []

    policy = policy_trainer.policy.to(device)
    policy.eval()

    refusals_so_far = 0
    overrides = 0
    t = 0

    for ex in examples:
        q_feats = compute_question_features(ex.question)
        bm_feats = compute_bm25_features(env.retriever, ex.question, top_k=5)
        extra_meta = {**q_feats, **bm_feats}

        # Build state
        state_np = state_encoder.encode(
            question=ex.question,
            domain_id=0,
            slo_vec=slo_vec,
            extra_meta=extra_meta,
        )
        state_tensor = torch.from_numpy(state_np).float().to(device)

        with torch.no_grad():
            action_id_tensor = policy.act(state_tensor)
        action_id = int(action_id_tensor.item())
        t += 1
        if max_refusal_frac is not None and action_id == 4:
            current_frac = refusals_so_far / max(1, t - 1)
            if current_frac >= max_refusal_frac:
                action_id = 0
                overrides += 1
            else:
                refusals_so_far += 1
        action_counts[action_id] += 1

        cfg = ACTIONS.get(action_id)
        if cfg is None or cfg.answer_mode == "refuse" or cfg.k == 0:
            retrieved_texts = []
        else:
            docs = env.retriever.retrieve(ex.question, top_k=cfg.k)
            retrieved_texts = [doc.get("text", "") for doc in docs]

        gold_answers = [ex.answer_text] if ex.answer_text else []
        answerable = any(ans.strip() for ans in gold_answers)
        hit = compute_retrieval_hit(gold_answers, retrieved_texts)
        answerable_flags.append(answerable)
        retrieval_hits.append(hit)

        step_result = env.step(
            question=ex.question,
            ground_truth=ex.answer_text,
            action_id=action_id,
            extra_eval=None,
        )

        metrics = compute_qa_score(
            answer=step_result.answer,
            ground_truth=ex.answer_text,
            extra_eval=None,
        )

        total_acc += float(metrics.get("accuracy", 0.0))
        total_cost += float(step_result.cost_tokens)
        total_halluc += int(metrics.get("hallucination", 0))
        # refusal if is_refusal() inside metrics (correct_refusal or wrong_refusal)
        total_refusal += int(metrics.get("correct_refusal", 0)) + int(metrics.get("wrong_refusal", 0))
        total_reward += float(step_result.reward)

        n += 1

    retrieval_hit_rate = compute_retrieval_hit_rate(answerable_flags, retrieval_hits)
    if n == 0:
        metrics = {
            "avg_accuracy": 0.0,
            "avg_cost_tokens": 0.0,
            "hallucination_rate": 0.0,
            "refusal_rate": 0.0,
            "avg_reward": 0.0,
            "retrieval_hit_rate": 0.0,
        }
    else:
        metrics = {
            "avg_accuracy": total_acc / n,
            "avg_cost_tokens": total_cost / n,
            "hallucination_rate": total_halluc / n,
            "refusal_rate": total_refusal / n,
            "avg_reward": total_reward / n,
            "retrieval_hit_rate": retrieval_hit_rate,
        }

    return metrics, dict(action_counts), overrides


def evaluate_fixed_action(
    action_id: int,
    examples,
    env: RagEnvironment,
    state_encoder: StateEncoder,
    slo_vec: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate a fixed action on a list of QA examples.

    Returns aggregated metrics:
      - avg_accuracy
      - avg_cost_tokens
      - hallucination_rate
      - refusal_rate
      - avg_reward
      - retrieval_hit_rate
    """
    from rl_rag_slo.llm_backend.answer_scorer import compute_qa_score as _compute

    total_acc = 0.0
    total_cost = 0.0
    total_halluc = 0
    total_refusal = 0
    total_reward = 0.0
    n = 0
    answerable_flags = []
    retrieval_hits = []

    cfg = ACTIONS.get(action_id)

    for ex in examples:
        q_feats = compute_question_features(ex.question)
        bm_feats = compute_bm25_features(env.retriever, ex.question, top_k=5)
        extra_meta = {**q_feats, **bm_feats}

        if cfg is None or cfg.answer_mode == "refuse" or cfg.k == 0:
            retrieved_texts = []
        else:
            docs = env.retriever.retrieve(ex.question, top_k=cfg.k)
            retrieved_texts = [doc.get("text", "") for doc in docs]

        gold_answers = [ex.answer_text] if ex.answer_text else []
        answerable = any(ans.strip() for ans in gold_answers)
        hit = compute_retrieval_hit(gold_answers, retrieved_texts)
        answerable_flags.append(answerable)
        retrieval_hits.append(hit)

        state_encoder.encode(
            question=ex.question,
            domain_id=0,
            slo_vec=slo_vec,
            extra_meta=extra_meta,
        )
        step_result = env.step(
            question=ex.question,
            ground_truth=ex.answer_text,
            action_id=action_id,
            extra_eval=None,
        )
        metrics = _compute(
            answer=step_result.answer,
            ground_truth=ex.answer_text,
            extra_eval=None,
        )
        total_acc += float(metrics.get("accuracy", 0.0))
        total_cost += float(step_result.cost_tokens)
        total_halluc += int(metrics.get("hallucination", 0))
        total_refusal += int(metrics.get("correct_refusal", 0)) + int(
            metrics.get("wrong_refusal", 0)
        )

        total_reward += float(step_result.reward)

        n += 1

    retrieval_hit_rate = compute_retrieval_hit_rate(answerable_flags, retrieval_hits)
    if n == 0:
        return {
            "avg_accuracy": 0.0,
            "avg_cost_tokens": 0.0,
            "hallucination_rate": 0.0,
            "refusal_rate": 0.0,
            "avg_reward": 0.0,
            "retrieval_hit_rate": 0.0,
        }

    return {
        "avg_accuracy": total_acc / n,
        "avg_cost_tokens": total_cost / n,
        "hallucination_rate": total_halluc / n,
        "refusal_rate": total_refusal / n,
        "avg_reward": total_reward / n,
        "retrieval_hit_rate": retrieval_hit_rate,
    }


def evaluate_baseline(
    examples,
    env: RagEnvironment,
    baseline_action_id: int,
) -> Dict[str, float]:
    """
    Evaluate a fixed baseline action on a list of QA examples.

    Returns aggregated metrics:
      - avg_accuracy
      - avg_cost_tokens
      - hallucination_rate
      - refusal_rate
      - avg_reward
      - retrieval_hit_rate
    """
    from rl_rag_slo.llm_backend.answer_scorer import compute_qa_score as _compute

    total_acc = 0.0
    total_cost = 0.0
    total_halluc = 0
    total_refusal = 0
    total_reward = 0.0
    n = 0
    answerable_flags = []
    retrieval_hits = []

    cfg = ACTIONS.get(baseline_action_id)

    for ex in examples:
        if cfg is None or cfg.answer_mode == "refuse" or cfg.k == 0:
            retrieved_texts = []
        else:
            docs = env.retriever.retrieve(ex.question, top_k=cfg.k)
            retrieved_texts = [doc.get("text", "") for doc in docs]

        gold_answers = [ex.answer_text] if ex.answer_text else []
        answerable = any(ans.strip() for ans in gold_answers)
        hit = compute_retrieval_hit(gold_answers, retrieved_texts)
        answerable_flags.append(answerable)
        retrieval_hits.append(hit)

        step_result = env.step(
            question=ex.question,
            ground_truth=ex.answer_text,
            action_id=baseline_action_id,
            extra_eval=None,
        )
        metrics = _compute(
            answer=step_result.answer,
            ground_truth=ex.answer_text,
            extra_eval=None,
        )
        total_acc += float(metrics.get("accuracy", 0.0))
        total_cost += float(step_result.cost_tokens)
        total_halluc += int(metrics.get("hallucination", 0))
        total_refusal += int(metrics.get("correct_refusal", 0)) + int(
            metrics.get("wrong_refusal", 0)
        )

        total_reward += float(step_result.reward)

        n += 1

    retrieval_hit_rate = compute_retrieval_hit_rate(answerable_flags, retrieval_hits)
    if n == 0:
        return {
            "avg_accuracy": 0.0,
            "avg_cost_tokens": 0.0,
            "hallucination_rate": 0.0,
            "refusal_rate": 0.0,
            "avg_reward": 0.0,
            "retrieval_hit_rate": 0.0,
        }

    return {
        "avg_accuracy": total_acc / n,
        "avg_cost_tokens": total_cost / n,
        "hallucination_rate": total_halluc / n,
        "refusal_rate": total_refusal / n,
        "avg_reward": total_reward / n,
        "retrieval_hit_rate": retrieval_hit_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RL RAG controller vs baseline on SQuAD 2.0."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained policy weights (policy.pt).",
    )
    parser.add_argument(
        "--squad_path",
        type=str,
        required=True,
        help="Path to SQuAD 2.0 JSON file.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1000,
        help="Number of examples to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run policy on: 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--slo_profile",
        type=str,
        default="quality_first",
        help="Name of SLO profile to use (e.g., 'quality_first', 'cheap').",
    )
    parser.add_argument(
        "--max_refusal_frac",
        type=float,
        default=None,
        help="Maximum allowed refusal fraction; overrides refusal action when exceeded.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to write evaluation outputs as JSON.",
    )
    args = parser.parse_args()

    examples = load_squad2_qa(args.squad_path)
    if args.num_examples > 0:
        examples = examples[: args.num_examples]

    corpus = build_squad2_corpus(examples)

    # llm_client = DummyLLMClient()
    retriever = BM25Retriever(corpus)
    llm_client = OpenAILLMClient()

    embedder = lambda q: deterministic_embed(q, dim=128)
    state_encoder = StateEncoder(embedder=embedder, num_domains=1)
    slo_vec = get_slo_vector(args.slo_profile)
    slo_weights = weights_for_profile(args.slo_profile)
    env = RagEnvironment(retriever=retriever, llm_client=llm_client, slo_weights=slo_weights)

    baseline_action_id = 1
    baseline_metrics = evaluate_baseline(examples, env, baseline_action_id)

    dummy_state = state_encoder.encode(
        question=examples[0].question if examples else "",
        domain_id=0,
        slo_vec=slo_vec,
        extra_meta=None,
    )
    state_dim = int(dummy_state.shape[0])
    num_actions = len(ACTIONS)

    trainer = BanditTrainer(
        state_dim=state_dim, num_actions=num_actions, lr=1e-3, device=args.device
    )
    trainer.load(args.model_path)

    learned_metrics, action_counts, refusal_overrides = evaluate_policy(
        examples=examples,
        env=env,
        state_encoder=state_encoder,
        slo_vec=slo_vec,
        policy_trainer=trainer,
        device=args.device,
        max_refusal_frac=args.max_refusal_frac,
    )

    print("=== Baseline (fixed action_id = 1) ===")
    for k, v in baseline_metrics.items():
        print(f"{k}: {v:.4f}")
    print("retrieval_hit_rate computed over answerable questions only")

    best_action_id = None
    best_metrics = None
    best_reward = None
    for action_id in sorted(ACTIONS.keys()):
        metrics = evaluate_fixed_action(
            action_id=action_id,
            examples=examples,
            env=env,
            state_encoder=state_encoder,
            slo_vec=slo_vec,
        )
        avg_reward = float(metrics.get("avg_reward", 0.0))
        if best_reward is None or avg_reward > best_reward:
            best_reward = avg_reward
            best_action_id = action_id
            best_metrics = metrics

    print("\n=== Best Fixed Action Baseline (by avg_reward) ===")
    if best_action_id is not None:
        cfg = ACTIONS.get(best_action_id)
        if cfg is not None:
            print(
                f"best_action_id={best_action_id} (k={cfg.k}, mode={cfg.answer_mode})"
            )
        else:
            print(f"best_action_id={best_action_id}")
    if best_metrics is not None:
        for k, v in best_metrics.items():
            print(f"{k}: {v:.4f}")
        print("retrieval_hit_rate computed over answerable questions only")

    print("\n=== Learned Policy ===")
    for k, v in learned_metrics.items():
        print(f"{k}: {v:.4f}")
    print("retrieval_hit_rate computed over answerable questions only")
    print("\n=== Learned Policy Action Distribution ===")
    total_actions = sum(action_counts.values()) or 1
    for action_id in sorted(action_counts.keys()):
        count = action_counts[action_id]
        frac = count / total_actions
        cfg = ACTIONS.get(action_id)
        if cfg is not None:
            print(
                f"action_id={action_id} (k={cfg.k}, mode={cfg.answer_mode}): "
                f"count={count}, frac={frac:.3f}"
            )
        else:
            print(
                f"action_id={action_id}: count={count}, frac={frac:.3f}"
            )
    print(f"\nrefusal_overrides: {refusal_overrides}")

    if args.output_json:
        baseline_cfg = ACTIONS.get(baseline_action_id)
        best_cfg = ACTIONS.get(best_action_id) if best_action_id is not None else None
        action_distribution = []
        for action_id in sorted(action_counts.keys()):
            cfg = ACTIONS.get(action_id)
            action_distribution.append(
                {
                    "action_id": int(action_id),
                    "k": int(cfg.k) if cfg is not None else None,
                    "mode": cfg.answer_mode if cfg is not None else None,
                    "count": int(action_counts[action_id]),
                    "frac": float(action_counts[action_id] / total_actions),
                }
            )

        output_payload = {
            "meta": {
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "model_path": args.model_path,
                "squad_path": args.squad_path,
                "num_examples": int(len(examples)),
                "device": args.device,
                "slo_profile": args.slo_profile,
                "max_refusal_frac": args.max_refusal_frac,
            },
            "baseline_fixed_action_id_1": {
                "action_id": int(baseline_action_id),
                "action_spec": {
                    "k": int(baseline_cfg.k) if baseline_cfg is not None else None,
                    "mode": baseline_cfg.answer_mode if baseline_cfg is not None else None,
                },
                "metrics": baseline_metrics,
            },
            "best_fixed_action_baseline": {
                "best_action_id": int(best_action_id) if best_action_id is not None else None,
                "action_spec": {
                    "k": int(best_cfg.k) if best_cfg is not None else None,
                    "mode": best_cfg.answer_mode if best_cfg is not None else None,
                },
                "metrics": best_metrics if best_metrics is not None else {},
            },
            "learned_policy": {
                "metrics": learned_metrics,
                "action_distribution": action_distribution,
                "refusal_overrides": int(refusal_overrides),
            },
        }
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, indent=2, sort_keys=False)


if __name__ == "__main__":
    main()
