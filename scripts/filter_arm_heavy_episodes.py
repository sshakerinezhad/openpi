"""Filter dataset to episodes with significant arm movement.

This script helps create a more balanced dataset by identifying and filtering
episodes based on the amount of arm movement vs base movement. Use this to
create a training set that forces the model to learn arm control.

Usage:
    python scripts/filter_arm_heavy_episodes.py --config-name=<your_config> --output-path=<path>
    
Example:
    python scripts/filter_arm_heavy_episodes.py \\
        --config-name=pi0_b1k \\
        --output-path=outputs/assets/pi05_b1k/arm_heavy_episodes.json \\
        --arm-ratio-threshold=1.5
"""

import json
import logging
import pathlib
from typing import Any

import numpy as np
import tyro
from tqdm import tqdm

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_episode_action_ratios(
    config: _config.TrainConfig,
    action_groups: dict[str, tuple[int, int]] | None = None,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    """Compute arm-to-base movement ratios for each episode.
    
    Args:
        config: Training configuration
        action_groups: Dictionary mapping group names to dimension ranges
        max_episodes: Maximum number of episodes to analyze
        
    Returns:
        Dictionary with episode statistics
    """
    # Default action groups for B1K
    if action_groups is None:
        action_groups = {
            "base": (0, 3),
            "trunk": (3, 7),
            "left_arm": (7, 14),
            "right_arm": (14, 21),
            "grippers": (21, 23),
        }
    
    logger.info("Creating data loader...")
    data_loader = _data_loader.create_data_loader(
        config,
        shuffle=False,
        framework="jax",
    )
    
    # Analyze episodes
    logger.info("Analyzing episodes...")
    episode_stats = []
    
    episode_idx = 0
    for observation, actions in tqdm(data_loader):
        if max_episodes is not None and episode_idx >= max_episodes:
            break
            
        # Assume each batch item is from a different episode
        # (this is a simplification - in reality you'd need episode boundaries)
        actions_np = np.array(actions)  # [B, H, D]
        
        for batch_idx in range(actions_np.shape[0]):
            episode_actions = actions_np[batch_idx]  # [H, D]
            
            # Compute delta actions
            delta_actions = episode_actions[1:] - episode_actions[:-1]  # [H-1, D]
            
            # Compute movement for each group
            group_movements = {}
            for group_name, (start_idx, end_idx) in action_groups.items():
                if start_idx >= episode_actions.shape[-1]:
                    continue
                end_idx = min(end_idx, episode_actions.shape[-1])
                
                group_deltas = delta_actions[:, start_idx:end_idx]
                group_movement = np.sum(np.linalg.norm(group_deltas, axis=-1))
                group_movements[group_name] = float(group_movement)
            
            # Compute arm-to-base ratio
            arm_movement = (
                group_movements.get("left_arm", 0) +
                group_movements.get("right_arm", 0)
            )
            base_movement = group_movements.get("base", 1e-6)
            arm_to_base_ratio = arm_movement / (base_movement + 1e-6)
            
            episode_stats.append({
                "episode_idx": episode_idx,
                "batch_idx": batch_idx,
                "arm_movement": arm_movement,
                "base_movement": base_movement,
                "trunk_movement": group_movements.get("trunk", 0),
                "gripper_movement": group_movements.get("grippers", 0),
                "arm_to_base_ratio": arm_to_base_ratio,
                "total_movement": sum(group_movements.values()),
            })
            
            episode_idx += 1
    
    logger.info(f"Analyzed {len(episode_stats)} episodes")
    
    return {
        "episodes": episode_stats,
        "num_episodes": len(episode_stats),
    }


def filter_episodes(
    episode_stats: dict[str, Any],
    arm_ratio_threshold: float = 1.5,
    min_arm_movement: float = 0.1,
    max_base_movement: float | None = None,
) -> dict[str, Any]:
    """Filter episodes based on arm/base movement criteria.
    
    Args:
        episode_stats: Statistics from compute_episode_action_ratios
        arm_ratio_threshold: Minimum arm-to-base movement ratio
        min_arm_movement: Minimum total arm movement
        max_base_movement: Maximum base movement (optional)
        
    Returns:
        Filtered episode statistics
    """
    episodes = episode_stats["episodes"]
    
    # Apply filters
    filtered_episodes = []
    for ep in episodes:
        if ep["arm_to_base_ratio"] >= arm_ratio_threshold:
            if ep["arm_movement"] >= min_arm_movement:
                if max_base_movement is None or ep["base_movement"] <= max_base_movement:
                    filtered_episodes.append(ep)
    
    logger.info(f"\n{'='*80}")
    logger.info("FILTERING RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Total episodes: {len(episodes)}")
    logger.info(f"Filtered episodes: {len(filtered_episodes)}")
    logger.info(f"Retention rate: {100 * len(filtered_episodes) / len(episodes):.1f}%")
    
    # Statistics
    arm_ratios = [ep["arm_to_base_ratio"] for ep in episodes]
    filtered_arm_ratios = [ep["arm_to_base_ratio"] for ep in filtered_episodes]
    
    logger.info(f"\nArm-to-base ratio statistics:")
    logger.info(f"  Original: mean={np.mean(arm_ratios):.2f}, median={np.median(arm_ratios):.2f}")
    if filtered_arm_ratios:
        logger.info(f"  Filtered: mean={np.mean(filtered_arm_ratios):.2f}, median={np.median(filtered_arm_ratios):.2f}")
    
    return {
        "episodes": filtered_episodes,
        "num_episodes": len(filtered_episodes),
        "filters": {
            "arm_ratio_threshold": arm_ratio_threshold,
            "min_arm_movement": min_arm_movement,
            "max_base_movement": max_base_movement,
        },
        "statistics": {
            "original_count": len(episodes),
            "filtered_count": len(filtered_episodes),
            "retention_rate": len(filtered_episodes) / len(episodes),
        },
    }


def main(
    config_name: str = "pi0_b1k",
    output_path: str = "outputs/assets/pi05_b1k/arm_heavy_episodes.json",
    arm_ratio_threshold: float = 1.5,
    min_arm_movement: float = 0.1,
    max_base_movement: float | None = None,
    max_episodes: int | None = 1000,
):
    """Filter dataset to arm-heavy episodes.
    
    Args:
        config_name: Name of the training config to use
        output_path: Path to save filtered episode indices
        arm_ratio_threshold: Minimum arm-to-base movement ratio
        min_arm_movement: Minimum total arm movement
        max_base_movement: Maximum base movement (optional, for very arm-focused training)
        max_episodes: Maximum number of episodes to analyze
    """
    # Load config
    config = _config.get_config(config_name)
    
    # Compute episode statistics
    episode_stats = compute_episode_action_ratios(
        config,
        max_episodes=max_episodes,
    )
    
    # Filter episodes
    filtered_stats = filter_episodes(
        episode_stats,
        arm_ratio_threshold=arm_ratio_threshold,
        min_arm_movement=min_arm_movement,
        max_base_movement=max_base_movement,
    )
    
    # Save results
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(filtered_stats, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Saved filtered episode list to: {output_path}")
    logger.info(f"{'='*80}")
    logger.info("\nTo use this filtered dataset in training:")
    logger.info("""
1. Modify your data loader to only load these episodes
2. Or use the episode indices to create a new dataset
3. Consider using this as a curriculum: train on arm-heavy episodes first,
   then fine-tune on full dataset
    """)


if __name__ == "__main__":
    tyro.cli(main)

