import json

import numpy as np

from src.openpi.models import tokenizer as _tokenizer

TASK_PROMPTS_PATH = "/vision/group/behavior/2025-challenge-demos/meta/tasks.jsonl"
with open(TASK_PROMPTS_PATH, "r") as f:
    task_prompts = [json.loads(line) for line in f]

tokenizer = _tokenizer.PaligemmaTokenizer(max_len=200)
for task_obj in task_prompts:
    task_index = task_obj["task_index"]
    task_name = task_obj["task_name"]
    prompt = task_obj["task"]
    tokens, masks = tokenizer.tokenize(f"[task-{task_index:02d}] {prompt}", state=np.random.rand(23), proprio_visibility_mask=np.random.randint(0, 2, size=23))
    if len(tokens) > 200:
        print(f"–––––– WARNING: idx: {task_index}, task_name: {task_name}, token length: {len(tokens)}")
    else:
        print(f"✓ idx: {task_index}, task_name: {task_name}, token length: {len(tokens)}")

breakpoint()
