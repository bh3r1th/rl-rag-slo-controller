import argparse
from typing import Dict, List

import numpy as np

from rl_rag_slo.datasets.squad2_loader import build_squad2_corpus, load_squad2_qa
from rl_rag_slo.retrievers.bm25_retriever import BM25Retriever
from rl_rag_slo.llm_backend.llm_client import OpenAILLMClient
from rl_rag_slo.controller.state_encoder import StateEncoder
from rl_rag_slo.slo.slo_config import get_slo_vector, weights_for_profile
from rl_rag_slo.controller.feature_utils import (
    compute_bm25_features,
    compute_question_features,
)
from rl_rag_slo.env.rag_env import RagEnvironment
from rl_rag_slo.controller.actions import ACTIONS
from rl_rag_slo.llm_backend.embedding_client import DeterministicHashEmbedder


def parse_slo_profiles(raw_profiles: str) -> List[str]:
    profiles = [p.strip() for p in raw_profiles.split(",")]
    return [p for p in profiles if p]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute offline replay buffer for RL RAG controller with multiple SLO profiles. "
            "For each question and profile, evaluate all actions in ACTIONS and store (state, action, reward)."
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
        "--slo_profiles",
        type=str,
        default="quality_first,cheap",
        help="Comma-separated list of SLO profiles (e.g., 'quality_first,cheap').",
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

    embedder = DeterministicHashEmbedder(dim=128)
    state_encoder = StateEncoder(embedder=embedder.embed, num_domains=1)

    slo_profiles = parse_slo_profiles(args.slo_profiles)
    if not slo_profiles:
        raise RuntimeError("No SLO profiles provided. Check --slo_profiles.")

    states_list = []
    actions_list = []
    rewards_list = []

    action_ids = sorted(ACTIONS.keys())
    printed_debug = False

    for idx, ex in enumerate(examples):
        question = ex.question
        ground_truth = ex.answer_text

        q_feats = compute_question_features(question)
        bm_feats = compute_bm25_features(retriever, question, top_k=5)
        extra_meta = {**q_feats, **bm_feats}

        for profile in slo_profiles:
            slo_vec = get_slo_vector(profile)
            slo_weights = weights_for_profile(profile)
            env = RagEnvironment(
                retriever=retriever,
                llm_client=llm_client,
                slo_weights=slo_weights,
            )

            state = state_encoder.encode(
                question=question,
                domain_id=0,
                slo_vec=slo_vec,
                extra_meta=extra_meta,
            )

            rewards_for_actions = []
            for action_id in action_ids:
                step_result = env.step(
                    question=question,
                    ground_truth=ground_truth,
                    action_id=action_id,
                    extra_eval=None,
                )
                rewards_for_actions.append(float(step_result.reward))

            if idx < 2:
                print(f"[debug] profile={profile} raw_rewards={rewards_for_actions}")

            mean_reward = float(np.mean(rewards_for_actions))
            for action_id, reward in zip(action_ids, rewards_for_actions):
                advantage = reward - mean_reward
                states_list.append(state)
                actions_list.append(int(action_id))
                rewards_list.append(float(advantage))

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} / {len(examples)} examples...")

    if not states_list:
        raise RuntimeError("No replay data generated. Something went wrong in precompute_replay_multi_slo.")

    states = np.stack(states_list).astype(np.float32)
    actions = np.asarray(actions_list, dtype=np.int64)
    rewards = np.asarray(rewards_list, dtype=np.float32)

    print("Using advantage normalization (reward - mean_reward_per_state) for training stability.")
    np.savez_compressed(
        args.output_path,
        states=states,
        actions=actions,
        rewards=rewards,
    )

    expected = len(examples) * len(slo_profiles) * len(action_ids)
    print(f"Saved replay buffer with {states.shape[0]} samples (expected {expected}) to {args.output_path}")


if __name__ == "__main__":
    main()
