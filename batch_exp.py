import os
data_root = "./data/wrench_data"
datasets = [
    # "youtube",
    # "sms",
    # "imdb",
    # "yelp",
    # "chemprot",
    "cdr",
    "agnews",
    "trec",
    # "medical_abstract"
]

samplers = [
    "passive",
    # "uncertain",
    # "QBC",
    # "weighted",
    # "SEU"
]

relation_extraction_datasets = ("chemprot", "cdr")
imbalanced_datasets = ("sms", "cdr")
multiclass_datasets = ("chemprot", "agnews", "trec", "medical_abstract")

acc = 0.6
agent = "chatgpt"
feature_extractor = "bert"

for dataset in datasets:
    for sampler in samplers:
        if dataset in relation_extraction_datasets:
            lf_type = "regex"
        else:
            lf_type = "keyword"

        if dataset in multiclass_datasets:
            lm_threshold = "fixed"
        else:
            lm_threshold = "auto"

        if dataset in imbalanced_datasets:
            metric = "f1"
        else:
            metric = "acc"

        # base prompt
        cmd = f"python main.py --dataset-path {data_root} --dataset-name {dataset} --feature-extractor {feature_extractor} " \
              f"--sampler {sampler} --lf-agent {agent} --lf-type {lf_type} --lf-acc-threshold {acc} --tune-metric {metric} " \
              f"--lm-threshold {lm_threshold} --max-ngram 3 --max-lf-per-iter 100 --display --save-wandb"
        if dataset == "cdr" and lf_type == "regex":
            cmd += " --default-class 0"
        # print(cmd)
        # os.system(cmd)
        # chain of thought
        cot_cmd = cmd + " --return-explanation"
        print(cot_cmd)
        os.system(cot_cmd)
        # # self-consistencey
        # sc_cmd = cot_cmd + " --n-completion 40"
        # print(sc_cmd)
        # os.system(sc_cmd)


