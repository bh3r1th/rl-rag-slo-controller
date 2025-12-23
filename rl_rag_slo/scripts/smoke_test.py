import argparse

import numpy as np

from rl_rag_slo.controller.actions import ACTIONS
from rl_rag_slo.controller.slo_profiles import get_slo_vector
from rl_rag_slo.controller.state_encoder import StateEncoder
from rl_rag_slo.datasets.squad2_loader import build_squad2_corpus, load_squad2_qa
from rl_rag_slo.env.rag_env import RagEnvironment
from rl_rag_slo.llm_backend.answer_scorer import compute_qa_score
from rl_rag_slo.llm_backend.llm_client import DummyLLMClient
from rl_rag_slo.retrievers.bm25_retriever import BM25Retriever


def deterministic_embed(question: str, dim: int = 128) -> np.ndarray:
    import hashlib

    h = hashlib.sha256(question.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little", signed=False)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for RL RAG environment with DummyLLMClient."
    )
    parser.add_argument(
        "--squad_path", type=str, required=True, help="Path to SQuAD 2.0 JSON file."
    )
    parser.add_argument(
        "--num_examples", type=int, default=20, help="Number of QA examples to test."
    )
    args = parser.parse_args()

    examples = load_squad2_qa(args.squad_path)
    examples = examples[: args.num_examples]

    corpus = build_squad2_corpus(examples)

    retriever = BM25Retriever(corpus)
    llm_client = DummyLLMClient()

    embedder = lambda q: deterministic_embed(q, dim=128)
    state_encoder = StateEncoder(embedder=embedder, num_domains=1)
    slo_vec = get_slo_vector("balanced")
    slo_weights = slo_vector_to_weights(slo_vec)
    env = RagEnvironment(retriever=retriever, llm_client=llm_client, slo_weights=slo_weights)

    action_ids = list(ACTIONS.keys())

    print(f"Running smoke test on {len(examples)} examples...")
    for idx, ex in enumerate(examples):
        state_encoder.encode(
            question=ex.question,
            domain_id=0,
            slo_vec=slo_vec,
            extra_meta=None,
        )

        action_id = action_ids[idx % len(action_ids)]

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

        print(f"\nExample {idx+1}:")
        print(f"  Q: {ex.question}")
        print(f"  Action id: {action_id}")
        print(
            f"  Reward: {step_result.reward:.4f}, Cost tokens: {step_result.cost_tokens}"
        )
        print(
            "  Metrics: accuracy={accuracy}, hallucination={hallucination}, "
            "correct_refusal={correct_refusal}, wrong_refusal={wrong_refusal}".format(
                **metrics
            )
        )

    print("\nSmoke test completed.")


if __name__ == "__main__":
    main()
