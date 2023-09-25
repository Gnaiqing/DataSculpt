import os
data_root = "./data/wrench_data"
datasets = [
    "youtube",
    "sms",
    "imdb",
    "yelp",
    "chemprot",
    "cdr"
]

relation_extraction_datasets = ("chemprot", "cdr")
imbalanced_datasets = ("sms", "cdr")
multiclass_datasets = ("chemprot",)

acc = 0.6
agent = "chatgpt"
feature_extractor = "bert"

for dataset in datasets:

    if dataset in relation_extraction_datasets:
        lf_types = ("regex",)
    else:
        lf_types = ("keyword",)

    if dataset in imbalanced_datasets:
        metric = "f1"
    else:
        metric = "acc"

    for lf_type in lf_types:
        # base prompt
        cmd = f"python main.py --dataset-path {data_root} --dataset-name {dataset} --feature-extractor {feature_extractor} " \
              f"--lf-agent {agent} --lf-type {lf_type} --lf-acc-threshold {acc} --tune-metric {metric} " \
              f"--max-ngram 3 --max-lf-per-iter 100 --display --save-wandb"
        if dataset == "cdr" and lf_type == "regex":
            cmd += " --default-class 0"
        print(cmd)
        os.system(cmd)
        # chain of thought
        cot_cmd = cmd + " --return-explanation"
        print(cot_cmd)
        os.system(cot_cmd)
        # # self-consistencey
        # sc_cmd = cot_cmd + " --n-completion 40"
        # print(sc_cmd)
        # os.system(sc_cmd)


