# Task Embeddings Implementation Guide

## Overview

Task embeddings have been added to provide explicit task conditioning for multi-task learning. This helps the model distinguish between different tasks during training and inference.

## What Was Added

### 1. **Model Changes** (`pi0.py`, `pi0_config.py`)
- Added `num_tasks` config parameter (default: 0, disabled)
- Added `task_embedding_scale` config parameter (default: 1.0)
- Added learnable task embedding layer (`nnx.Embed`) when `num_tasks > 0`
- Task embeddings are added to the time conditioning signal in the diffusion model

### 2. **Data Pipeline** (`model.py`, `transforms.py`, `data_loader.py`)
- Added `task_id` field to `Observation` dataclass
- Added `ExtractTaskID` transform that extracts task_id from task_index
- Task ID is automatically extracted from dataset and passed through the pipeline

### 3. **Parameter Cost**
- For BEHAVIOR (50 tasks) with 300M action expert: **~102K parameters**
  - Calculation: `50 tasks × 2048 hidden_dim = 102,400 parameters`
- Negligible overhead (<0.04% of total model size)

## How to Enable

### Stage 1: Multi-Task Foundation Training (22 tasks)

Add to your config or command line:

```bash
--model.num_tasks=50  # Total number of BEHAVIOR tasks
--model.task_embedding_scale=1.0  # Default scale
```

Example command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run scripts/train_val.py pi05_b1k_22_TASKS_oversample \
    --exp_name="foundation_22tasks_with_task_emb" \
    --overwrite \
    --batch_size=256 \
    --model.num_tasks=50 \
    --model.task_embedding_scale=1.0 \
    --weight_loader.params_path=gs://openpi-assets/checkpoints/pi05_base/params \
    --num_train_steps=100000 \
    --val_log_interval=3000
```

### Stage 2: Task-Specific Fine-Tuning

When fine-tuning on a single task, keep the same `num_tasks=50` setting:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run scripts/train_val.py pi05_b1k_single_task \
    --exp_name="finetune_task_01" \
    --overwrite \
    --batch_size=128 \
    --model.num_tasks=50 \
    --model.task_embedding_scale=1.0 \
    --weight_loader.params_path=/path/to/stage1/checkpoint \
    --optimizer.learning_rate=5e-6 \  # 10x lower than stage 1!
    --num_train_steps=10000 \
    --val_log_interval=1000
```

## How It Works

1. **Data Loading**: `task_index` from dataset → `ExtractTaskID` transform → `task_id` in batch
2. **Forward Pass**: 
   - Time embedding: `time_emb = posemb_sincos(timestep, ...)`
   - Task conditioning: `time_emb += task_embedding_scale * task_embeddings(task_id)`
   - Diffusion: Task-conditioned time_emb → adaRMS normalization → action generation

3. **Effect**: Task embeddings directly modulate the action expert's layer normalization, providing a strong task-specific signal alongside the diffusion timestep.

## Configuration Options

### `num_tasks` (int, default: 0)
- Set to 0 to disable task embeddings (backward compatible)
- Set to 50 for BEHAVIOR competition (or total number of tasks in your dataset)
- Must match the range of task_index values in your dataset

### `task_embedding_scale` (float, default: 1.0)
- Scale factor for task embedding contribution to time conditioning
- Default 1.0 treats task and time embeddings equally
- Increase to >1.0 if task confusion persists
- Decrease to <1.0 if model overfits to task IDs

## Tuning Guide

### If You See Task Confusion (Model Mixing Tasks):
1. Increase `task_embedding_scale` to 1.5 or 2.0
2. Check that task prefixes in prompts are still present (`[01]` in text)
3. Verify task_id is being extracted correctly (check data loader output)

### If Model Ignores Task Variations Within Same Task:
1. Decrease `task_embedding_scale` to 0.5
2. This forces the model to rely more on language prompts for nuance

### If Model Doesn't Generalize to New Tasks:
1. Train with `num_tasks=0` (disabled) to rely purely on language
2. Or train on more diverse tasks in stage 1

## Technical Details

### Architecture Integration
```python
# In Pi0.embed_suffix():
time_emb = posemb_sincos(timestep, embedding_dim, ...)

if self.num_tasks > 0 and obs.task_id is not None:
    task_emb = self.task_embeddings(obs.task_id)  # [B, emb]
    time_emb = time_emb + self.task_embedding_scale * task_emb

# time_emb then conditions action expert via adaRMS or MLP
```

### Backward Compatibility
- Default `num_tasks=0` disables task embeddings
- Existing configs work without modification
- Old checkpoints can be loaded (task_embeddings are new params)

### Data Format
```python
# Input batch should contain:
{
    "task_index": 5,  # Integer task ID from dataset
    "task_id": np.array([5], dtype=np.int32),  # Automatically added by ExtractTaskID
    # ... other fields
}
```

## Troubleshooting

### Error: "task_id not found in batch"
- Check that your dataset includes `task_index` field
- Verify `ExtractTaskID` transform is in the pipeline

### Error: "task_id out of range"
- Set `num_tasks` to at least `max(task_index) + 1`
- For BEHAVIOR: use `num_tasks=50`

### Task embeddings not loading from checkpoint
- This is expected when loading from a checkpoint trained without task embeddings
- New task embedding parameters will be randomly initialized
- Fine-tune for a few thousand steps to learn good task embeddings

## Performance Impact

- **Training speed**: Negligible (<1% overhead)
- **Memory**: +102K parameters (0.04% of Pi0.5 model)
- **Inference**: No impact on sampling speed

## Comparison with Alternatives

| Method | Params | Strength | When to Use |
|--------|--------|----------|-------------|
| Text prefix only | 0 | Moderate | Single/few tasks, strong LLM |
| Task embeddings | ~100K | Strong | Multi-task, systematic confusion |
| Task LoRA heads | ~50K per task | Very Strong | Extreme task differences |
| Full task heads | ~3M | Strongest | Different action spaces |

## Recommended Strategy

1. **Start**: Text prefixes only (your current setup)
2. **If task confusion**: Add task embeddings (`num_tasks=50`, `scale=1.0`)
3. **If still confused**: Increase scale to 1.5-2.0
4. **Last resort**: Task-specific LoRA (not yet implemented)

---

**Questions?** The implementation is in:
- Config: `src/openpi/models/pi0_config.py`
- Model: `src/openpi/models/pi0.py`
- Transform: `src/openpi/transforms.py` (`ExtractTaskID`)

