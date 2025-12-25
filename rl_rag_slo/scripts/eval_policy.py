import argparse
from typing import Dict

import numpy as np
import torch

from rl_rag_slo.datasets.squad2_loader import load_squad2_qa, build_squad2_corpus
from rl_rag_slo.retrievers.bm25_retriever import BM25Retriever
from rl_rag_slo.llm_backend.llm_client import DummyLLMClient, OpenAILLMClient
from rl_rag_slo.controller.state_encoder import StateEncoder
from rl_rag_slo.controller.slo_profiles import get_slo_vector
from rl_rag_slo.env.rag_env import RagEnvironment
from rl_rag_slo.controller.bandit_trainer import BanditTrainer
from rl_rag_slo.controller.actions import ACTIONS
from rl_rag_slo.llm_backend.answer_scorer import compute_qa_score


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


def slo_vector_to_weights(vec: np.ndarray) -> dict:
    v = vec.astype(np.float32)
    return {
        "w_quality": float(v[0]),
        "w_cost": float(v[1]),
        "w_refusal": float(v[2]),
        "w_halluc": float(v[3]),
    }


def evaluate_policy(
    examples,
    env: RagEnvironment,
    state_encoder: StateEncoder,
    slo_vec: np.ndarray,
    policy_trainer: BanditTrainer,
    device: str,
) -> Dict[str, float]:
    """
    Evaluate the learned policy on a list of QA examples.

    Returns aggregated metrics:
      - avg_accuracy        (EM-based, mostly for reference with real LLMs)
      - avg_cost_tokens
      - hallucination_rate
      - refusal_rate
      - avg_reward          (environment reward)
      - retrieval_hit_rate  (fraction of examples where retrieval_hit == 1)
    """
    total_acc = 0.0
    total_cost = 0.0
    total_halluc = 0
    total_refusal = 0
    total_reward = 0.0
    total_retrieval_hit = 0
    n = 0

    policy = policy_trainer.policy.to(device)
    policy.eval()

    for ex in examples:
        # Build state
        state_np = state_encoder.encode(
            question=ex.question,
            domain_id=0,
            slo_vec=slo_vec,
            extra_meta=None,
        )
        state_tensor = torch.from_numpy(state_np).float().to(device)

        with torch.no_grad():
            action_id_tensor = policy.act(state_tensor)
        action_id = int(action_id_tensor.item())

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
        total_refusal += int(metrics.get("correct_refusal", 0)) + int(
            metrics.get("wrong_refusal", 0)
        )

        total_reward += float(step_result.reward)
        total_retrieval_hit += int(step_result.meta.get("retrieval_hit", 0))

        n += 1

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
        "retrieval_hit_rate": total_retrieval_hit / n,
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
    total_retrieval_hit = 0
    n = 0

    for ex in examples:
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
        total_retrieval_hit += int(step_result.meta.get("retrieval_hit", 0))

        n += 1

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
        "retrieval_hit_rate": total_retrieval_hit / n,
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
    slo_vec = get_slo_vector("quality_first")
    slo_weights = slo_vector_to_weights(slo_vec)
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

    learned_metrics = evaluate_policy(
        examples=examples,
        env=env,
        state_encoder=state_encoder,
        slo_vec=slo_vec,
        policy_trainer=trainer,
        device=args.device,
    )

    print("=== Baseline (fixed action_id = 1) ===")
    for k, v in baseline_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Learned Policy ===")
    for k, v in learned_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
