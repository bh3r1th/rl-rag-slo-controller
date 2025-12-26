import argparse

from rl_rag_slo.datasets.squad2_loader import load_squad2_qa, build_squad2_corpus
from rl_rag_slo.retrievers.bm25_retriever import BM25Retriever
from rl_rag_slo.llm_backend.llm_client import OpenAILLMClient
from rl_rag_slo.slo.slo_config import get_slo_vector, weights_for_profile
from rl_rag_slo.env.rag_env import RagEnvironment
from rl_rag_slo.controller.actions import ACTIONS
from rl_rag_slo.llm_backend.embedding_client import DeterministicHashEmbedder
from rl_rag_slo.controller.state_encoder import StateEncoder
from rl_rag_slo.llm_backend.answer_scorer import compute_qa_score


def evaluate_fixed_action(examples, env: RagEnvironment, action_id: int):
    total_acc = 0.0
    total_cost = 0.0
    total_halluc = 0
    total_refusal = 0
    total_reward = 0.0
    n = 0

    for ex in examples:
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
        total_refusal += int(metrics.get("correct_refusal", 0)) + int(metrics.get("wrong_refusal", 0))
        total_reward += float(step_result.reward)
        n += 1

    if n == 0:
        return {
            "avg_accuracy": 0.0,
            "avg_cost_tokens": 0.0,
            "hallucination_rate": 0.0,
            "refusal_rate": 0.0,
            "avg_reward": 0.0,
        }

    return {
        "avg_accuracy": total_acc / n,
        "avg_cost_tokens": total_cost / n,
        "hallucination_rate": total_halluc / n,
        "refusal_rate": total_refusal / n,
        "avg_reward": total_reward / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect per-action metrics for RL RAG controller on SQuAD v2.0."
    )
    parser.add_argument(
        "--squad_path",
        type=str,
        required=True,
        help="Path to SQuAD 2.0 JSON (dev or train).",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=200,
        help="Number of QA examples to use.",
    )
    parser.add_argument(
        "--slo_profile",
        type=str,
        default="quality_first",
        help="SLO profile name (e.g., 'quality_first', 'cheap').",
    )
    args = parser.parse_args()

    examples = load_squad2_qa(args.squad_path)
    if args.num_examples > 0:
        examples = examples[: args.num_examples]

    if not examples:
        raise RuntimeError("No SQuAD examples loaded. Check squad_path and num_examples.")

    corpus = build_squad2_corpus(examples)
    retriever = BM25Retriever(corpus)
    llm_client = OpenAILLMClient()

    slo_vec = get_slo_vector(args.slo_profile)
    slo_weights = weights_for_profile(args.slo_profile)
    env = RagEnvironment(retriever=retriever, llm_client=llm_client, slo_weights=slo_weights)

    print(f"Evaluating {len(examples)} examples under SLO profile '{args.slo_profile}'")
    for action_id, cfg in sorted(ACTIONS.items()):
        metrics = evaluate_fixed_action(examples, env, action_id)
        print(f"\n=== Action {action_id} (k={cfg.k}, mode={cfg.answer_mode}) ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
