import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
    parser.add_argument(
        "--objective",
        type=str,
        choices=["reward_weighted", "argmax_ce"],
        default="reward_weighted",
        help="Training objective: reward_weighted or argmax_ce.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5.0,
        help="Margin weighting factor for argmax_ce objective.",
    )
    args = parser.parse_args()

    data = np.load(args.replay_path)
    states_np = data["states"].astype(np.float32)
    actions_np = data["actions"].astype(np.int64)
    rewards_np = data["rewards"].astype(np.float32)
    print(f"Loaded replay: N={states_np.shape[0]}, state_dim={states_np.shape[1]}")

    state_dim = int(states_np.shape[1])
    # Use the current ACTIONS definition as the source of truth.
    num_actions = len(ACTIONS)

    if args.objective == "reward_weighted":
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
    else:
        print("Training objective: argmax_ce (supervised best-action classification)")
        n = states_np.shape[0]
        block_size = 5
        if n % block_size != 0:
            trimmed = (n // block_size) * block_size
            print(f"Warning: N={n} not divisible by {block_size}; truncating to {trimmed}.")
            states_np = states_np[:trimmed]
            actions_np = actions_np[:trimmed]
            rewards_np = rewards_np[:trimmed]
            n = trimmed
        if n == 0:
            raise RuntimeError("No samples to train after truncation.")

        b = n // block_size
        x_block = states_np[0:n:5]
        a_block = actions_np.reshape(b, block_size)
        r_block = rewards_np.reshape(b, block_size)

        state_dim = int(x_block.shape[1])
        slo_len = 4
        extra_len = 11
        slo_start = state_dim - (slo_len + extra_len)
        slo_end = slo_start + slo_len
        slo = x_block[:, slo_start:slo_end]
        is_quality = (slo[:, 0] > slo[:, 1]) & (slo[:, 0] > slo[:, 2])
        is_cheap = ~is_quality

        best_before = a_block[np.arange(b), np.argmax(r_block, axis=1)]
        count_refuse_best_before = int(np.sum((best_before == 4) & is_cheap))

        mask = (a_block == 4) & is_cheap[:, None]
        r_masked = r_block.copy()
        r_masked[mask] = -1e9

        best_j = np.argmax(r_masked, axis=1)
        best_val = r_masked[np.arange(b), best_j]
        second_best_val = np.partition(r_masked, -2, axis=1)[:, -2]
        margin = (best_val - second_best_val).astype(np.float32)
        best_action = a_block[np.arange(b), best_j]

        print(
            "Margin stats: "
            f"mean={float(np.mean(margin)):.6f}, "
            f"p90={float(np.percentile(margin, 90)):.6f}, "
            f"max={float(np.max(margin)):.6f}"
        )

        print(f"Blocks: B={b}, state_dim={state_dim}, num_actions={num_actions}")
        print(
            f"Cheap blocks: {int(np.sum(is_cheap))}, "
            f"Quality blocks: {int(np.sum(is_quality))}"
        )
        print(
            "Cheap blocks where refuse would be best BEFORE masking: "
            f"{count_refuse_best_before}"
        )

        device = torch.device(args.device)
        trainer = BanditTrainer(
            state_dim=state_dim,
            num_actions=num_actions,
            lr=1e-3,
            device=args.device,
        )
        policy = trainer.policy
        optimizer = trainer.optimizer
        weights = 1.0 + args.alpha * margin
        weights = np.clip(weights, 0.0, 10.0).astype(np.float32)
        criterion = nn.CrossEntropyLoss(reduction="none")

        states_tensor = torch.from_numpy(x_block).to(device)
        best_action_tensor = torch.from_numpy(best_action.astype(np.int64)).to(device)
        weights_tensor = torch.from_numpy(weights).to(device)

        dataset = TensorDataset(states_tensor, best_action_tensor, weights_tensor)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        for epoch in range(1, args.epochs + 1):
            total_loss = 0.0
            total_n = 0
            for batch_states, batch_actions, batch_weights in loader:
                logits = policy(batch_states)
                loss_vec = criterion(logits, batch_actions)
                loss = (loss_vec * batch_weights).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size_actual = batch_states.size(0)
                total_loss += loss.item() * batch_size_actual
                total_n += batch_size_actual

            avg_loss = total_loss / total_n if total_n > 0 else 0.0
            print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.6f}")

        trainer.save(args.output_path)
    print(f"Saved trained policy to {args.output_path}")


if __name__ == "__main__":
    main()
