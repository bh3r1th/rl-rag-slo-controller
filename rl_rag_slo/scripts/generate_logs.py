import argparse
import hashlib
from typing import Tuple

import numpy as np

from rl_rag_slo.controller.actions import ACTIONS
from rl_rag_slo.slo.slo_config import get_slo_vector, slo_vector_to_weights
from rl_rag_slo.controller.state_encoder import StateEncoder
from rl_rag_slo.datasets.squad2_loader import QAExample, build_squad2_corpus, load_squad2_qa
from rl_rag_slo.env.rag_env import RagEnvironment
from rl_rag_slo.llm_backend.llm_client import DummyLLMClient
from rl_rag_slo.retrievers.bm25_retriever import BM25Retriever


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline replay buffer for RL RAG controller using SQuAD 2.0."
    )
    parser.add_argument(
        "--squad_path", type=str, required=True, help="Path to SQuAD 2.0 JSON file."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output .npz replay file."
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5000,
        help="Maximum number of QA examples to sample.",
    )
    args = parser.parse_args()

    # 1) Load SQuAD examples and build corpus
    examples = load_squad2_qa(args.squad_path)
    if args.num_examples > 0:
        examples = examples[: args.num_examples]

    corpus = build_squad2_corpus(examples)

    # 2) Build retriever and LLM client
    retriever = BM25Retriever(corpus)
    llm_client = DummyLLMClient()

    # 3) Build StateEncoder with deterministic embedder
    embedder = lambda q: deterministic_embed(q, dim=128)
    state_encoder = StateEncoder(embedder=embedder, num_domains=1)

    # 4) Build SLO weights (use "balanced" profile)
    slo_vec = get_slo_vector("quality_first")
    slo_weights = slo_vector_to_weights(slo_vec)
    env = RagEnvironment(retriever=retriever, llm_client=llm_client, slo_weights=slo_weights)

    # 5) Generate replay data
    states_list = []
    actions_list = []
    rewards_list = []

    action_ids = list(ACTIONS.keys())

    for idx, ex in enumerate(examples):
        # Encode state
        state = state_encoder.encode(
            question=ex.question,
            domain_id=0,
            slo_vec=slo_vec,
            extra_meta=None,
        )

        # Sample an action uniformly at random
        action_id = int(np.random.choice(action_ids))

        # Step environment
        step_result = env.step(
            question=ex.question,
            ground_truth=ex.answer_text,
            action_id=action_id,
            extra_eval=None,
        )

        states_list.append(state)
        actions_list.append(action_id)
        rewards_list.append(step_result.reward)

    if not states_list:
        raise RuntimeError("No replay data generated. Check dataset and configuration.")

    states = np.stack(states_list).astype(np.float32)
    actions = np.array(actions_list, dtype=np.int64)
    rewards = np.array(rewards_list, dtype=np.float32)

    np.savez_compressed(
        args.output_path,
        states=states,
        actions=actions,
        rewards=rewards,
    )
    print(f"Saved replay buffer with {states.shape[0]} samples to {args.output_path}")


if __name__ == "__main__":
    main()
