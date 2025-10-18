import json

import torch
import torch.nn.functional as F

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    tor_only_path = "outputs/assets/pi0_b1k/behavior-1k/2025-challenge-demos/norm_stats_pi0_b1k_tor.json"
    all_task_path = "outputs/assets/pi0_b1k/behavior-1k/2025-challenge-demos/norm_stats_pi0_b1k_all_tasks.json"
    tor_only_norm_stats = load_json(tor_only_path)
    all_task_norm_stats = load_json(all_task_path)
    tor_only_actions_mean = tor_only_norm_stats["norm_stats"]["actions"]["mean"]
    all_task_actions_mean = all_task_norm_stats["norm_stats"]["actions"]["mean"]
    print("MSE loss between tor only and all task actions mean: ", F.mse_loss(torch.tensor(tor_only_actions_mean), torch.tensor(all_task_actions_mean)))
    print("Cosine similarity between tor only and all task actions mean: ", F.cosine_similarity(torch.tensor(tor_only_actions_mean), torch.tensor(all_task_actions_mean), dim=0))

if __name__ == "__main__":
    main()
