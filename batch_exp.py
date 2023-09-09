import os
data_root = "./data/wrench_data"
datasets = [
    "youtube",
    # "sms",
    # "imdb",
    # "yelp",
    # "chemprot",
    # "cdr"
]

relation_extraction_datasets = ("chemprot", "cdr")

lf_accs = [0.6]
lf_agent_types = ["chatgpt"]

for dataset in datasets:
    if dataset in relation_extraction_datasets:
        feature_extractor = "bert"
    else:
        feature_extractor = "tfidf"

    for acc in lf_accs:
        for agent in lf_agent_types:
            cmd = f"python main.py --dataset-path {data_root} --dataset-name {dataset} --feature-extractor {feature_extractor} " \
                  f"--lf-agent {agent} --lf-acc-threshold {acc} --lf-filter acc unique --display --save-wandb"
            print(cmd)
            os.system(cmd)