import os
data_root = "./data"
datasets = [
    # "youtube",
    "sms",
    # "IMDB",
    # "Yelp",
    # "AmazonReview"
]
lf_accs = [0.6]
lf_agent_types = ["chatgpt"]
prompt_versions = ["v1", "v2"]

for dataset in datasets:
    for acc in lf_accs:
        for agent in lf_agent_types:
            for version in prompt_versions:
                cmd = f"python main.py --dataset-path {data_root} --dataset-name {dataset} " \
                      f"--lf-agent {agent} --lf-acc-threshold {acc} --lf-filter unique --llm-prompt-version {version} " \
                      f"--display --save-wandb"
                print(cmd)
                os.system(cmd)