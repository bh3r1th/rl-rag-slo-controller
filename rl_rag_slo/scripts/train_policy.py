import argparse

import numpy as np
import torch

from rl_rag_slo.controller.actions import ACTIONS
from rl_rag_slo.controller.bandit_trainer import BanditBatch, BanditTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RL RAG controller from offline replay buffer."
    )
    parser.add_argument(
        "--replay_path",
        type=str,
        required=True,
        help="Path to .npz replay file (states, actions, rewards).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save trained policy weights (e.g., policy.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: 'cpu' or 'cuda'.",
    )
    args = parser.parse_args()

    data = np.load(args.replay_path)
    states_np = data["states"].astype(np.float32)
    actions_np = data["actions"].astype(np.int64)
    rewards_np = data["rewards"].astype(np.float32)

    state_dim = int(states_np.shape[1])
    num_actions = max(int(actions_np.max()) + 1, len(ACTIONS))

    states = torch.from_numpy(states_np)
    actions = torch.from_numpy(actions_np)
    rewards = torch.from_numpy(rewards_np)

    replay = BanditBatch(states=states, actions=actions, rewards=rewards)
    trainer = BanditTrainer(
        state_dim=state_dim,
        num_actions=num_actions,
        lr=1e-3,
        device=args.device,
    )

    for epoch in range(1, args.epochs + 1):
        loss = trainer.train_epoch(replay, batch_size=args.batch_size)
        print(f"Epoch {epoch}/{args.epochs} - loss: {loss:.6f}")

    trainer.save(args.output_path)
    print(f"Saved trained policy to {args.output_path}")


if __name__ == "__main__":
    main()
