from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from oran_scheduler.rl.ppo_update import (
    PPOOptimizers,
)


def save_ppo_model_checkpoint(
    *,
    path: str | Path,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizers: PPOOptimizers,
    tti_index: int,
    num_ppo_updates: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Save model/optimizer state for analysis and later
    frozen-policy evaluation.

    IMPORTANT:
        This is NOT a complete simulator-state
        checkpoint.

        Traffic queues, PF history, radio RNG state,
        rollout buffers and unresolved boundaries are
        not captured here.

        Therefore it must not be described as an
        exact mid-run resume mechanism.
    """

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "tti_index": int(
            tti_index
        ),
        "num_ppo_updates": int(
            num_ppo_updates
        ),
        "actor_state_dict": (
            actor.state_dict()
        ),
        "critic_state_dict": (
            critic.state_dict()
        ),
        "actor_optimizer_state_dict": (
            optimizers.actor.state_dict()
        ),
        "critic_optimizer_state_dict": (
            optimizers.critic.state_dict()
        ),
        "metadata": (
            {}
            if metadata is None
            else dict(
                metadata
            )
        ),
    }

    torch.save(
        payload,
        output_path,
    )

    return output_path


def load_ppo_model_checkpoint(
    *,
    path: str | Path,
    actor: torch.nn.Module,
    critic: torch.nn.Module | None = None,
    optimizers: PPOOptimizers | None = None,
    map_location: (
        str | torch.device
    ) = "cpu",
) -> dict[str, Any]:
    """
    Load a saved PPO model checkpoint.

    actor is always restored.

    critic / optimizers are optional so the same
    checkpoint can be used for frozen actor evaluation.
    """

    checkpoint = torch.load(
        Path(
            path
        ),
        map_location=(
            map_location
        ),
        weights_only=False,
    )

    actor.load_state_dict(
        checkpoint[
            "actor_state_dict"
        ]
    )

    if critic is not None:
        critic.load_state_dict(
            checkpoint[
                "critic_state_dict"
            ]
        )

    if optimizers is not None:
        optimizers.actor.load_state_dict(
            checkpoint[
                "actor_optimizer_state_dict"
            ]
        )

        optimizers.critic.load_state_dict(
            checkpoint[
                "critic_optimizer_state_dict"
            ]
        )

    return checkpoint


