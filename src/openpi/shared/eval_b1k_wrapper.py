import numpy as np
import torch
import os
import json
from openpi.training import config as _config
from openpi_client.base_policy import BasePolicy
from openpi_client.image_tools import resize_with_pad
from collections import deque
import copy
import traceback

RESIZE_SIZE = 224

class B1KPolicyWrapper():
    def __init__(
        self, 
        policy: BasePolicy,
        config: _config.TrainConfig,
        text_prompt: str = "Turn on the radio receiver that's on the table in the living room.",
        control_mode: str = "temporal_ensemble",
        max_actions_per_pred: int = 128,   # Truncate each prediction sequence to this length (UNUSED)
        replan_interval: int = 1,          # Steps between replanning (1=every step, 10=every 10, 50=rarely)
        max_predictions: int = 10,         # Max predictions to keep for averaging
        exp_k_value: float = 0.005,        # Exponential weighting factor (higher = favor recent more)
    ) -> None:
        self.policy = policy
        self.text_prompt = text_prompt
        self.control_mode = control_mode

        if control_mode == "receeding_horizon":
            assert max_predictions == 1, "If you are using receeding_horizon, you only take 1 prediction at a time, there is no ensembling (averaging) of predictions"
        elif control_mode == "temporal_ensemble":
            assert replan_interval == 1, "If you are using temporal_ensemble, you need to replan every step"
            assert max_predictions > 1, "If you are using temporal_ensemble, you need to keep at least 2 predictions for ensembling, otherwise this is just receeding_horizon with predicting 1 step at a time"
        elif control_mode == "receeding_temporal":
            assert replan_interval > 1, "If you are using receeding_temporal, you need to replan more than one step at a time, otherwise this is just temporal_ensemble"
            assert max_predictions > 1, "If you are using receeding_temporal, you need to keep at least 2 predictions for ensembling, otherwise this is just receeding_horizon"
        else:
            raise ValueError(f"Invalid control mode: {control_mode}")

        # Core parameters
        self.replan_interval = replan_interval
        self.max_predictions = max_predictions
        self.max_actions_per_pred = max_actions_per_pred
        self.exp_k_value = exp_k_value
        self.step_counter = 0

        # Queue of prediction sequences (each is a deque of actions)
        # maxlen automatically drops oldest when full
        self.prediction_queue = deque([], maxlen=max_predictions)

        # For error recovery
        self.last_action = {"actions": np.zeros((max_actions_per_pred, 23), dtype=np.float64)}

        dataset_root = config.data.base_config.behavior_dataset_root
        self.task_prompt_map = {}
        TASKS_METADATA_PATH = os.path.join(dataset_root, "meta/tasks.jsonl")
        if os.path.exists(TASKS_METADATA_PATH):
            with open(TASKS_METADATA_PATH, "r") as f:
                for line in f:
                    task = json.loads(line)
                    self.task_prompt_map[task["task_index"]] = task["task"]

    def reset(self):
        self.prediction_queue = deque([], maxlen=self.max_predictions)
        self.last_action = {"actions": np.zeros((self.max_actions_per_pred, 23), dtype=np.float64)}
        self.step_counter = 0

    def get_prompt_from_obs(self, obs: dict) -> str:
        if "prompt" in obs:
            return obs["prompt"]

        task_id = int(obs["task_id"])
        if task_id in self.task_prompt_map:
            return self.task_prompt_map[task_id]

        return self.text_prompt

    def process_obs(self, obs: dict) -> dict:
        """
        Process the observation dictionary to match the expected input format for the model.
        """
        prop_state = obs["robot_r1::proprio"][None]
        img_obs = np.stack(
            [
                resize_with_pad(
                    obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE
                ),
                resize_with_pad(
                    obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][None, ..., :3], 
                    RESIZE_SIZE,
                    RESIZE_SIZE
                ),
                resize_with_pad(
                    obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE
                ),
            ],
            axis=1,
        )
        processed_obs = {
            "observation": img_obs,  # Shape: (1, 3, H, W, C)
            "proprio": prop_state,
            "prompt": self.get_prompt_from_obs(obs),
            "task_index": int(obs["task_id"]),  # read it in as task_index because that's what _transforms.ExtractTaskID() expects
        }
        return processed_obs
    
    def _prepare_batch(self, input_obs):
        """Extract and format observation for policy inference"""
        nbatch = copy.deepcopy(input_obs)
        if nbatch["observation"].shape[-1] != 3:
            nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

        joint_positions = nbatch["proprio"][0]
        return {
            "observation/egocentric_camera": nbatch["observation"][0, 0],
            "observation/wrist_image_left": nbatch["observation"][0, 1],
            "observation/wrist_image_right": nbatch["observation"][0, 2],
            "observation/state": joint_positions,
            "prompt": input_obs["prompt"],
            "task_index": input_obs["task_index"],
        }

    def _act_temporal_ensemble(self, input_obs):
        """
        Unified implementation that handles all three modes:
        - receeding_horizon: replan_interval=max_actions_per_pred, max_predictions=1
        - receeding_temporal: replan_interval=10, max_predictions=5
        - temporal_ensemble: replan_interval=1, max_predictions=10
        """
        # Step 1: Check if we should replan
        should_replan = (self.step_counter % self.replan_interval == 0)

        if should_replan:
            # Run policy inference
            batch = self._prepare_batch(input_obs)

            try:
                action = self.policy.infer(batch)
                self.last_action = action
            except Exception as e:
                action = self.last_action
                print(f"Error in action prediction, using last action: {traceback.format_exc()}")
                # raise e

            # Truncate and store as new prediction sequence
            target_actions = action["actions"][:self.max_actions_per_pred].copy()
            new_prediction = deque(target_actions)
            self.prediction_queue.append(new_prediction)  # Automatically drops oldest if full

        # Step 2: Extract current action from all stored predictions
        if len(self.prediction_queue) == 0:
            raise ValueError("Prediction queue is empty - this shouldn't happen!")

        # Get next action from each prediction
        actions_to_average = []
        for pred_seq in self.prediction_queue:
            if len(pred_seq) > 0:
                actions_to_average.append(pred_seq.popleft())

        # Remove exhausted predictions
        self.prediction_queue = deque(
            [seq for seq in self.prediction_queue if len(seq) > 0],
            maxlen=self.max_predictions
        )

        # Step 3: Average with exponential weighting
        if len(actions_to_average) == 1:
            # No averaging needed
            final_action = actions_to_average[0]
        else:
            actions_array = np.array(actions_to_average)

            # Compute exponential weights (older=lower weight, newer=higher weight)
            exp_weights = np.exp(self.exp_k_value * np.arange(len(actions_to_average)))
            exp_weights = exp_weights / exp_weights.sum()

            # Weighted average
            final_action = (actions_array * exp_weights[:, None]).sum(axis=0)

            # Override gripper channels with newest prediction
            final_action[-9] = actions_to_average[-1][-9]
            final_action[-1] = actions_to_average[-1][-1]

        self.step_counter += 1
        return final_action[None]

    def act(self, input_obs):
        """
        Main action selection method. Uses unified temporal ensemble algorithm.
        
        Model input expected: 
            📌 Key: observation/exterior_image_1_left
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: observation/exterior_image_2_left
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: observation/joint_position
            Type: ndarray
            Dtype: float64
            Shape: (16,)

            📌 Key: prompt
            Type: str
            Value: do something
        
        Model will output:
            📌 Key: actions
            Type: ndarray
            Dtype: float64
            Shape: (10, 16)
        """
        # Process observation
        input_obs = self.process_obs(input_obs)

        # Fast path for receeding_horizon to avoid unnecessary processing
        if (
            self.replan_interval >= self.max_actions_per_pred and 
            self.max_predictions == 1 and
            len(self.prediction_queue) > 0 and 
            len(self.prediction_queue[0]) > 0
        ):
            # Still have actions in queue, just pop and return
            action = self.prediction_queue[0].popleft()
            self.step_counter += 1
            return torch.from_numpy(action[None])

        # General path: unified temporal ensemble
        final_action = self._act_temporal_ensemble(input_obs)
        return torch.from_numpy(final_action)
