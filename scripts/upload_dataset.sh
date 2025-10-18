#!/bin/bash

TASK_NAMES=("task-0000" "task-0001" "task-0023" "task-0035" "task-0047")

echo "This is UPLOADING dataset, not downloading!!! Tasks=${TASK_NAMES[*]}"

for TASK_NAME in "${TASK_NAMES[@]}"; do
    for DIR in annotations data skill_prompts videos meta/episodes; do
        echo "Syncing ${DIR} for ${TASK_NAME}..."
        aws s3 sync \
            /vision/group/behavior/2025-challenge-demos/${DIR}/${TASK_NAME}/ \
            s3://behavior-challenge/vision/group/behavior/2025-challenge-demos/${DIR}/${TASK_NAME}/
    done
done

# This is done separately because it is not task-specific. In fact, it is redundant except for the
# very first task that is uploaded. (Which has already happened, but we'll keep this for completeness.)
aws s3 sync --exclude "episodes/*" \
    /vision/group/behavior/2025-challenge-demos/meta/ \
    s3://behavior-challenge/vision/group/behavior/2025-challenge-demos/meta/
