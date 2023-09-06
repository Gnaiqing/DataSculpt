import os
data_root = "./data/wrench_data"
datasets = [
    # "youtube",
    # "sms",
    # "imdb",
    # "yelp",
    "chemprot"
]

lf_accs = [0.6]
lf_agent_types = ["simulated", "chatgpt"]

for dataset in datasets:
    for acc in lf_accs:
        for agent in lf_agent_types:
            cmd = f"python main.py --dataset-path {data_root} --dataset-name {dataset} --feature-extractor bert " \
                  f"--lf-agent {agent} --lf-acc-threshold {acc} --lf-filter acc unique " \
                  f"--display --save-wandb"
            print(cmd)
            os.system(cmd)