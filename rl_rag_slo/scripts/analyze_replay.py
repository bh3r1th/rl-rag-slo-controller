import argparse
from typing import Dict, List, Tuple

import numpy as np


def infer_profile_label(slo_slice: np.ndarray) -> str:
    if slo_slice[0] > slo_slice[1] and slo_slice[0] > slo_slice[2]:
        return "quality_first"
    return "cheap"


def format_counts_and_fracs(counts: Dict[int, int], total: int, action_ids: List[int]) -> List[str]:
    lines = []
    for action_id in action_ids:
        count = counts.get(action_id, 0)
        frac = count / total if total > 0 else 0.0
        lines.append(f"  action_id={action_id}: count={count}, frac={frac:.3f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze multi-SLO replay buffers and summarize per-SLO action preferences."
    )
    parser.add_argument(
        "--replay_path",
        type=str,
        required=True,
        help="Path to replay .npz file (states, actions, rewards).",
    )
    args = parser.parse_args()

    data = np.load(args.replay_path)
    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]

    n = states.shape[0]
    block_size = 5
    if n % block_size != 0:
        trimmed = (n // block_size) * block_size
        print(f"Warning: N={n} not divisible by {block_size}; truncating to {trimmed}.")
        states = states[:trimmed]
        actions = actions[:trimmed]
        rewards = rewards[:trimmed]
        n = trimmed

    if n == 0:
        raise RuntimeError("No samples to analyze after truncation.")

    slo_start = 144 - (4 + 11)
    slo_end = slo_start + 4

    action_ids = sorted(set(int(a) for a in actions.tolist()))
    overall_best_counts: Dict[int, int] = {a: 0 for a in action_ids}
    profile_best_counts: Dict[str, Dict[int, int]] = {
        "quality_first": {a: 0 for a in action_ids},
        "cheap": {a: 0 for a in action_ids},
    }
    profile_action_counts: Dict[str, Dict[int, int]] = {
        "quality_first": {a: 0 for a in action_ids},
        "cheap": {a: 0 for a in action_ids},
    }
    profile_action_rewards: Dict[str, Dict[int, List[float]]] = {
        "quality_first": {a: [] for a in action_ids},
        "cheap": {a: [] for a in action_ids},
    }

    num_blocks = n // block_size
    for block_idx in range(num_blocks):
        i = block_idx * block_size
        slo_slice = states[i, slo_start:slo_end]
        if block_idx < 3:
            print(f"[slo] block={block_idx} slice={slo_slice.tolist()}")

        profile = infer_profile_label(slo_slice)
        block_rewards = rewards[i : i + block_size]
        best_offset = int(np.argmax(block_rewards))
        best_action_id = int(actions[i + best_offset])
        overall_best_counts[best_action_id] += 1
        profile_best_counts[profile][best_action_id] += 1

        for j in range(block_size):
            action_id = int(actions[i + j])
            profile_action_counts[profile][action_id] += 1
            profile_action_rewards[profile][action_id].append(float(rewards[i + j]))

    print("\nOverall best-action distribution:")
    for line in format_counts_and_fracs(overall_best_counts, num_blocks, action_ids):
        print(line)

    for profile in ["quality_first", "cheap"]:
        profile_blocks = sum(profile_best_counts[profile].values())
        print(f"\nProfile: {profile}")
        print("Best-action counts/fractions:")
        for line in format_counts_and_fracs(profile_best_counts[profile], profile_blocks, action_ids):
            print(line)
        print("Average reward per action_id:")
        for action_id in action_ids:
            rewards_list = profile_action_rewards[profile][action_id]
            avg_reward = float(np.mean(rewards_list)) if rewards_list else 0.0
            count = profile_action_counts[profile][action_id]
            print(f"  action_id={action_id}: avg_reward={avg_reward:.3f}, count={count}")


if __name__ == "__main__":
    main()
