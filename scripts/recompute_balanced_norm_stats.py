"""Recompute normalization statistics with per-group balancing.

The standard normalization may inadvertently cause arm/navigation imbalance
if the action groups have very different scales. This script recomputes
normalization stats with per-group standardization to ensure all action
groups are on similar scales.

Usage:
    python scripts/recompute_balanced_norm_stats.py --config-name=<your_config> --output-dir=<path>
    
Example:
    python scripts/recompute_balanced_norm_stats.py \\
        --config-name=pi0_b1k \\
        --output-dir=outputs/assets/pi05_b1k/behavior-1k/balanced
"""

import logging
import pathlib
from typing import Any

import numpy as np
import tyro
from tqdm import tqdm

from openpi import transforms as _transforms
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_per_group_norm_stats(
    config: _config.TrainConfig,
    max_frames: int | None = None,
    action_groups: dict[str, tuple[int, int]] | None = None,
    balance_groups: bool = True,
) -> dict[str, _transforms.NormStats]:
    """Compute normalization statistics with optional per-group balancing.
    
    Args:
        config: Training configuration
        max_frames: Maximum number of frames to use
        action_groups: Dictionary mapping group names to dimension ranges
        balance_groups: If True, scale each group to have similar variance
        
    Returns:
        Dictionary containing normalization statistics
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
        skip_norm_stats=True,  # Don't apply normalization
        framework="jax",
    )
    
    # Collect data
    logger.info("Collecting data for normalization statistics...")
    all_states = []
    all_actions = []
    
    total_frames = 0
    for observation, actions in tqdm(data_loader):
        batch_states = np.array(observation.state)  # [B, D]
        batch_actions = np.array(actions)  # [B, H, D]
        
        # Flatten action horizon
        batch_actions = batch_actions.reshape(-1, batch_actions.shape[-1])  # [B*H, D]
        
        all_states.append(batch_states)
        all_actions.append(batch_actions)
        
        total_frames += batch_states.shape[0]
        if max_frames is not None and total_frames >= max_frames:
            break
    
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    
    logger.info(f"Collected {len(all_states)} state samples and {len(all_actions)} action samples")
    
    # Compute standard statistics
    logger.info("Computing standard normalization statistics...")
    state_stats = _transforms.NormStats(
        mean=np.mean(all_states, axis=0),
        std=np.std(all_states, axis=0),
        q01=np.percentile(all_states, 1, axis=0),
        q99=np.percentile(all_states, 99, axis=0),
    )
    
    action_stats = _transforms.NormStats(
        mean=np.mean(all_actions, axis=0),
        std=np.std(all_actions, axis=0),
        q01=np.percentile(all_actions, 1, axis=0),
        q99=np.percentile(all_actions, 99, axis=0),
    )
    
    if not balance_groups:
        return {"state": state_stats, "actions": action_stats}
    
    # Apply per-group balancing
    logger.info("\n" + "="*80)
    logger.info("APPLYING PER-GROUP BALANCING")
    logger.info("="*80)
    
    action_dim = all_actions.shape[-1]
    balanced_action_std = np.copy(action_stats.std)
    balanced_action_q_range = action_stats.q99 - action_stats.q01
    
    # Compute target scale (median of group stds)
    group_stds = []
    for group_name, (start_idx, end_idx) in action_groups.items():
        if start_idx >= action_dim:
            continue
        end_idx = min(end_idx, action_dim)
        group_std = np.mean(action_stats.std[start_idx:end_idx])
        group_stds.append(group_std)
    
    target_std = np.median(group_stds)
    logger.info(f"\nTarget std (median of groups): {target_std:.4f}")
    
    # Scale each group to have similar variance
    logger.info("\nGroup adjustments:")
    for group_name, (start_idx, end_idx) in action_groups.items():
        if start_idx >= action_dim:
            continue
        end_idx = min(end_idx, action_dim)
        
        group_std = np.mean(action_stats.std[start_idx:end_idx])
        scale_factor = target_std / (group_std + 1e-8)
        
        # Apply scaling
        balanced_action_std[start_idx:end_idx] = action_stats.std[start_idx:end_idx] * scale_factor
        balanced_action_q_range[start_idx:end_idx] = (
            action_stats.q99[start_idx:end_idx] - action_stats.q01[start_idx:end_idx]
        ) * scale_factor
        
        logger.info(f"  {group_name:15s}: original_std={group_std:.4f}, "
                   f"scale_factor={scale_factor:.2f}, "
                   f"new_std={np.mean(balanced_action_std[start_idx:end_idx]):.4f}")
    
    # Update q01 and q99 to match new scale
    balanced_action_q01 = action_stats.mean - balanced_action_q_range / 2
    balanced_action_q99 = action_stats.mean + balanced_action_q_range / 2
    
    balanced_action_stats = _transforms.NormStats(
        mean=action_stats.mean,
        std=balanced_action_std,
        q01=balanced_action_q01,
        q99=balanced_action_q99,
    )
    
    logger.info("\nBalanced normalization will equalize the importance of different action groups.")
    
    return {"state": state_stats, "actions": balanced_action_stats}


def main(
    config_name: str = "pi0_b1k",
    output_dir: str | None = None,
    max_frames: int | None = 100000,
    balance_groups: bool = True,
):
    """Recompute normalization statistics with optional per-group balancing.
    
    Args:
        config_name: Name of the training config to use
        output_dir: Output directory for norm stats (if None, uses default)
        max_frames: Maximum number of frames to use (None = all)
        balance_groups: Whether to balance action groups to similar scales
    """
    # Load config
    config = _config.get_config(config_name)
    
    # Compute statistics
    norm_stats = compute_per_group_norm_stats(
        config,
        max_frames=max_frames,
        balance_groups=balance_groups,
    )
    
    # Determine output directory
    if output_dir is None:
        data_config = config.data.create(
            pathlib.Path(config.assets_base_dir),
            config.model,
        )
        if data_config.asset_id is not None:
            output_dir = (
                pathlib.Path(config.assets_base_dir) /
                config.exp_name /
                data_config.repo_id /
                "balanced"
            )
        else:
            output_dir = pathlib.Path("outputs/assets") / config.exp_name / "balanced"
    else:
        output_dir = pathlib.Path(output_dir)
    
    # Save statistics
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "norm_stats.json"
    
    _transforms.Normalize.save(output_dir, norm_stats)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Saved balanced normalization statistics to: {output_path}")
    logger.info(f"{'='*80}")
    logger.info("\nTo use these statistics in training, add this to your config:")
    logger.info(f"""
    data=LeRobotB1KDataConfig(
        ...,
        assets=AssetsConfig(
            assets_dir="{output_dir.parent}",
            asset_id="balanced",
        ),
    )
    """)


if __name__ == "__main__":
    tyro.cli(main)

