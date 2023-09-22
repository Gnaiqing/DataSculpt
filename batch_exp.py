import os
data_root = "./data/wrench_data"
datasets = [
    # "youtube",
    # "sms",
    # "imdb",
    # "yelp",
    # "chemprot",
    "cdr"
]

relation_extraction_datasets = ("chemprot", "cdr")
imbalanced_datasets = ("sms", "cdr")

acc = 0.6
agent = "chatgpt"

for dataset in datasets:
    if dataset in relation_extraction_datasets:
        feature_extractor = "bert"
    else:
        feature_extractor = "tfidf"

    if dataset in imbalanced_datasets:
        metric = "f1"
    else:
        metric = "acc"

    # base prompt
    cmd = f"python main.py --dataset-path {data_root} --dataset-name {dataset} --feature-extractor {feature_extractor} " \
          f"--lf-agent {agent} --lf-acc-threshold {acc} --tune-metric {metric} --max-ngram 3 --max-lf-per-iter 100 --display --save-wandb"
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


