# Quick Start: Fix Arm Joint Stagnation

Your policy is great at navigation but sucks at using arms? Here's the fastest way to fix it.

## TL;DR - Run This Now 🚀

```bash
# Option 1: Use the pre-configured arm-focused training
python scripts/train.py --config-name=pi0_b1k_arm_focused

# Option 2: For PyTorch training
python scripts/train_pytorch.py --config-name=pi0_b1k_arm_focused
```

That's it! This uses **per-group loss weighting** to give arms 3x more importance than navigation.

---

## What Changed?

The fix modifies how the loss function weights different action groups:

**Before**: Navigation movements (large) dominated the loss → model ignored arms (small movements)

**After**: Arms get 3x weight → model learns both navigation AND arm control

---

## If It's Still Not Working

### Try Stronger Arm Emphasis

Edit your config or use the "very focused" version:

```bash
python scripts/train.py --config-name=pi0_b1k_very_arm_focused
```

This gives arms 5x importance (vs 3x in the default fix).

### Or Adjust Manually

In your training config:

```python
model=pi0_config.Pi0Config(
    # ... other settings ...
    loss_weighting_strategy="per_group",
    group_weights={
        "base": 1.0,
        "trunk": 2.0,
        "left_arm": 5.0,   # Try 4.0, 5.0, or even 10.0 if needed
        "right_arm": 5.0,
        "grippers": 3.0,
    },
)
```

---

## Verify It's Working

### During Training

Check your WandB/logs for:
- ✅ Loss decreasing (not just staying flat after initial drop)
- ✅ Arm-related metrics improving

### After Training

Test on manipulation tasks:
```bash
python scripts/eval_policy.py --checkpoint=<path> --tasks=pick_and_place,drawer_opening
```

You should see the robot actually moving its arms now!

---

## Advanced: Analyze Your Data First

Want to understand the problem before applying the fix?

```bash
# See action magnitude imbalance
python scripts/analyze_action_magnitudes.py --config-name=pi0_b1k

# This shows you:
# - How much bigger navigation movements are vs arm movements
# - Suggested weights for balancing
```

---

## Full Documentation

See [ARM_STAGNATION_SOLUTIONS.md](ARM_STAGNATION_SOLUTIONS.md) for:
- Complete technical explanation
- Multiple solution approaches
- Data filtering and normalization tools
- Troubleshooting guide

---

## What If I Have a Different Robot?

You'll need to adjust the action groups. Example for your robot:

```python
action_groups={
    "base": (0, 2),        # Your base action dims
    "arm": (2, 9),         # Your arm action dims
    "gripper": (9, 10),    # Your gripper dims
}

group_weights={
    "base": 1.0,
    "arm": 3.0,    # Emphasize arm
    "gripper": 2.0,
}
```

---

## Still Having Issues?

1. Make sure your normalization stats are reasonable (no NaN or extreme values)
2. Try starting with `loss_weighting_strategy="uniform"` as a baseline
3. Check that your action space dimensions are correct
4. Open a GitHub issue with your config and training logs

---

## Summary

- **Problem**: Navigation dominates loss → arms ignored
- **Solution**: Weight arms higher in loss (3-5x)
- **How**: Use `loss_weighting_strategy="per_group"` in config
- **Result**: Model learns both navigation AND arm control ✨

