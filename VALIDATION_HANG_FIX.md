# Multi-GPU Training Hang Fix

## Problem
The training script was hanging after approximately 5361 steps in multi-GPU setups. When resumed from the checkpoint at 5000 steps, it would hang again at 10362 steps (another ~5362 steps later). This pattern indicated the issue occurred at regular intervals corresponding to the validation step frequency.

## Root Cause
The issue was caused by improper cleanup of validation data loaders in multi-GPU environments:

1. **Validation runs periodically** (every `val_log_interval` steps, ~5000 steps)
2. **Each validation creates a new data loader** with underlying PyTorch DataLoader objects
3. **No explicit cleanup** of these data loaders after validation completes
4. **Resource leaks accumulate** in multi-GPU setups, causing hangs when iterators and distributed state aren't properly released

Even with `num_workers=0` (set to prevent deadlock, see line 124), the PyTorch DataLoader still maintains internal state that needs proper cleanup in distributed training environments.

## Solution
Added explicit cleanup of validation data loaders and iterators:

### Changes Made:

1. **Added `gc` module import** (line 3)
   - Enables forced garbage collection to ensure resource cleanup

2. **Iterator cleanup in `_compute_validation_losses()`** (lines 173-189)
   ```python
   try:
       for batch_idx in range(config.val_num_batches):
           # ... validation logic ...
   finally:
       # Clean up the iterator to prevent resource leaks
       del val_iter
   ```

3. **Data loader cleanup in `compute_validation_loss()`** (lines 228-243)
   ```python
   try:
       val_loss = _compute_validation_losses(...)
   finally:
       # Explicitly clean up the validation data loader
       if hasattr(val_loader, '_torch_data_loader'):
           torch_data_loader = val_loader._torch_data_loader
           if hasattr(torch_data_loader, '_data_loader'):
               del torch_data_loader._data_loader
       del val_loader
       # Force garbage collection
       gc.collect()
   ```

## Why This Works
- **Explicit deletion** ensures Python releases references to PyTorch objects
- **Forced garbage collection** (`gc.collect()`) immediately frees resources instead of waiting for automatic GC
- **try-finally blocks** guarantee cleanup even if validation fails
- **Nested cleanup** handles both the iterator and the data loader itself

## Testing
After applying this fix:
- Training should continue past the previous hang points (5361, 10362 steps)
- No resource accumulation during validation
- Safe cleanup even if validation encounters errors

## Related Code Locations
- `/workspace/openpi/scripts/train_val.py` - Main training script with validation
- `/workspace/openpi/src/openpi/training/data_loader.py` - TorchDataLoader implementation
- Line 124: Comment acknowledging multi-GPU deadlock issues with num_workers
- Line 472: Validation trigger point in training loop

