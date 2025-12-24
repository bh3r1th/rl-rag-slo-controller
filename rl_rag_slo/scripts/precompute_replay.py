import argparse
from typing import Dict

import numpy as np

from rl_rag_slo.datasets.squad2_loader import load_squad2_qa, build_squad2_corpus
from rl_rag_slo.retrievers.bm25_retriever import BM25Retriever
from rl_rag_slo.llm_backend.llm_client import OpenAILLMClient
from rl_rag_slo.controller.state_encoder import StateEncoder
from rl_rag_slo.controller.slo_profiles import get_slo_vector
from rl_rag_slo.env.rag_env import RagEnvironment
from rl_rag_slo.controller.actions import ACTIONS
from rl_rag_slo.llm_backend.embedding_client import DeterministicHashEmbedder


def slo_vector_to_weights(vec: np.ndarray) -> Dict[str, float]:
    """
    Convert a 4D SLO vector [w_quality, w_cost, w_refusal, w_halluc]
    into a dict with additional lambda parameters.
    """
    v = vec.astype(np.float32)
    return {
        "w_quality": float(v[0]),
        "w_cost": float(v[1]),
        "w_refusal": float(v[2]),
        "w_halluc": float(v[3]),
        # Reasonable defaults; can be tuned per experiment
        "lambda_cost": 1e-3,
        "lambda_halluc": 1.0,
        "lambda_wrong_ref": 1.0,
        "lambda_correct_ref": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute offline replay buffer for RL RAG controller using SQuAD 2.0 and gpt-4.1-nano. "
            "For each question, evaluate all actions in ACTIONS and store (state, action, reward)."
        )
    )
    parser.add_argument(
        "--squad_path",
        type=str,
        required=True,
        help="Path to SQuAD 2.0 JSON file (train or dev).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output .npz replay file.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1000,
        help="Maximum number of QA examples to use.",
    )
    parser.add_argument(
        "--slo_profile",
        type=str,
        default="quality_first",
        help="Name of SLO profile to use (e.g., 'quality_first', 'cheap').",
    )
    args = parser.parse_args()

    # 1) Load SQuAD examples
    examples = load_squad2_qa(args.squad_path)
    if args.num_examples > 0:
        examples = examples[: args.num_examples]

    if not examples:
        raise RuntimeError("No SQuAD examples loaded. Check squad_path and num_examples.")

    # 2) Build corpus and retriever
    corpus = build_squad2_corpus(examples)
    retriever = BM25Retriever(corpus)

    # 3) LLM client (gpt-4.1-nano via OpenAILLMClient)
    llm_client = OpenAILLMClient()

    # 4) State encoder with deterministic hash embedder to avoid embedding API cost for now.
    #    For a stronger model, you can switch to OpenAIEmbedder later.
    embedder = DeterministicHashEmbedder(dim=128)
    state_encoder = StateEncoder(embedder=embedder.embed, num_domains=1)

    # 5) SLO profile and environment
    slo_vec = get_slo_vector(args.slo_profile)
    slo_weights = slo_vector_to_weights(slo_vec)
    env = RagEnvironment(retriever=retriever, llm_client=llm_client, slo_weights=slo_weights)

    # 6) Build replay: for each example and each action, store (state, action_id, reward)
    states_list = []
    actions_list = []
    rewards_list = []

    action_ids = sorted(ACTIONS.keys())

    for idx, ex in enumerate(examples):
        question = ex.question
        ground_truth = ex.answer_text

        # Encode state once per question (domain_id=0 for SQuAD)
        state = state_encoder.encode(
            question=question,
            domain_id=0,
            slo_vec=slo_vec,
            extra_meta=None,
        )

        for action_id in action_ids:
            step_result = env.step(
                question=question,
                ground_truth=ground_truth,
                action_id=action_id,
                extra_eval=None,
            )
            states_list.append(state)
            actions_list.append(int(action_id))
            rewards_list.append(float(step_result.reward))

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} / {len(examples)} examples...")

    if not states_list:
        raise RuntimeError("No replay data generated. Something went wrong in precompute_replay.")

    states = np.stack(states_list).astype(np.float32)
    actions = np.asarray(actions_list, dtype=np.int64)
    rewards = np.asarray(rewards_list, dtype=np.float32)

    np.savez_compressed(
        args.output_path,
        states=states,
        actions=actions,
        rewards=rewards,
    )
    print(f"Saved replay buffer with {states.shape[0]} samples to {args.output_path}")


if __name__ == "__main__":
    main()
