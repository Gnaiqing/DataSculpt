import os
data_root = "./data"
datasets = [
    "youtube",
    # "sms",
    # "IMDB",
    # "Yelp",
    # "AmazonReview"
]
lf_accs = [0.6]
lf_agent_types = ["simulated"]

for dataset in datasets:
    for acc in lf_accs:
        for agent in lf_agent_types:
            cmd = f"python main.py --dataset-path {data_root} --dataset-name {dataset} " \
                  f"--lf-agent {agent} --lf-acc-threshold {acc} " \
                  f"--save-wandb --use-soft-labels"
            print(cmd)
            os.system(cmd)