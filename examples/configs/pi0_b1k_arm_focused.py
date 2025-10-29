"""Example training configuration for B1K with arm-focused loss weighting.

This configuration addresses the issue where the policy learns navigation well
but struggles with arm control. It uses per-group loss weighting to emphasize
arm joints over base navigation.

Usage:
    python scripts/train.py --config examples/configs/pi0_b1k_arm_focused.py
"""

import dataclasses
from openpi.training import config as _config
from openpi.models import pi0_config


# Example 1: Use per-group weighting with high arm emphasis (RECOMMENDED)
@dataclasses.dataclass(frozen=True)
class Pi0B1KArmFocusedConfig(_config.TrainConfig):
    """Training config that emphasizes arm control over navigation."""
    
    exp_name: str = "pi0_b1k_arm_focused"
    
    model: pi0_config.Pi0Config = dataclasses.field(
        default_factory=lambda: pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=50,
            pi05=True,  # Use Pi0.5 architecture
            
            # Configure loss weighting to emphasize arms
            loss_weighting_strategy="per_group",  # Use per-group weighting
            
            # Define action groups for B1K robot (23 dims padded to 32)
            action_groups={
                "base": (0, 3),        # base x-y-theta velocity
                "trunk": (3, 7),       # trunk joints
                "left_arm": (7, 14),   # left arm 7 joints
                "right_arm": (14, 21), # right arm 7 joints
                "grippers": (21, 23),  # gripper widths
                "padding": (23, 32),   # padding dimensions
            },
            
            # Group weights: Higher = more important in loss
            # Default was: base=1.0, arms=1.0 (implicit equal weighting)
            # New: Emphasize arms 3x more than base
            group_weights={
                "base": 1.0,
                "trunk": 2.0,
                "left_arm": 3.0,   # 3x emphasis on arms
                "right_arm": 3.0,  # 3x emphasis on arms
                "grippers": 2.5,
                "padding": 0.0,    # No loss on padding
            },
        )
    )
    
    # Use B1K data configuration
    data: _config.DataConfigFactory = dataclasses.field(
        default_factory=lambda: _config.LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            # Optionally filter for arm-heavy tasks
            # tasks=["pick_and_place", "drawer_opening", "button_press"],
        )
    )
    
    batch_size: int = 256
    num_train_steps: int = 50000
    

# Example 2: Even more aggressive arm focus
@dataclasses.dataclass(frozen=True)
class Pi0B1KVeryArmFocusedConfig(Pi0B1KArmFocusedConfig):
    """Very aggressive arm emphasis - use if arms are still struggling."""
    
    exp_name: str = "pi0_b1k_very_arm_focused"
    
    model: pi0_config.Pi0Config = dataclasses.field(
        default_factory=lambda: pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=50,
            pi05=True,
            loss_weighting_strategy="per_group",
            action_groups={
                "base": (0, 3),
                "trunk": (3, 7),
                "left_arm": (7, 14),
                "right_arm": (14, 21),
                "grippers": (21, 23),
                "padding": (23, 32),
            },
            group_weights={
                "base": 0.5,       # Reduce base importance
                "trunk": 2.0,
                "left_arm": 5.0,   # 10x more than base!
                "right_arm": 5.0,
                "grippers": 3.0,
                "padding": 0.0,
            },
        )
    )


# Example 3: Simple per-dimension weighting (easier to tune)
@dataclasses.dataclass(frozen=True)
class Pi0B1KSimpleArmFocusedConfig(_config.TrainConfig):
    """Simpler approach using fixed per-dimension weights."""
    
    exp_name: str = "pi0_b1k_simple_arm_focused"
    
    model: pi0_config.Pi0Config = dataclasses.field(
        default_factory=lambda: pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=50,
            pi05=True,
            loss_weighting_strategy="per_dimension",  # Simpler approach
        )
    )
    

# Example 4: Uniform weighting (baseline comparison)
@dataclasses.dataclass(frozen=True)
class Pi0B1KUniformConfig(_config.TrainConfig):
    """Baseline with uniform weighting across all dimensions."""
    
    exp_name: str = "pi0_b1k_uniform"
    
    model: pi0_config.Pi0Config = dataclasses.field(
        default_factory=lambda: pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=50,
            pi05=True,
            loss_weighting_strategy="uniform",  # No weighting
        )
    )

