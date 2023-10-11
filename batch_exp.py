import os
data_root = "./data/wrench_data"
datasets = [
    # "youtube",
    # "sms",
    # "imdb",
    # "yelp",
    # "agnews",
    # "trec",
    # "medical_abstract",
    "arxiv_abstract"
    # "chemprot",
    # "cdr",
    # "spouse",
    # "semeval"
]

samplers = [
    "passive",
    # "uncertain",
    # "QBC",
    # "weighted",
    # "SEU"
]

label_models = ["Snorkel", "MV"]

relation_extraction_datasets = ("chemprot", "cdr", "spouse", "semeval")
imbalanced_datasets = ("sms", "cdr", "spouse")
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

        if dataset in imbalanced_datasets:
            metric = "f1"
        else:
            metric = "acc"

        # base prompt
        cmd = f"python main.py --dataset-path {data_root} --dataset-name {dataset} --feature-extractor {feature_extractor} " \
              f"--sampler {sampler} --lf-agent {agent} --lf-type {lf_type} --lf-acc-threshold {acc} --tune-metric {metric} " \
              f"--label-model Snorkel --max-ngram 3 --max-lf-per-iter 100 --display --save-wandb"
        # cmd += " --example-selection neighbor"
        if dataset in ["cdr", "spouse"]:
            cmd += " --default-class 0"
        # print(cmd)
        # os.system(cmd)
        # chain of thought
        cot_cmd = cmd + " --return-explanation"
        print(cot_cmd)
        os.system(cot_cmd)
        # # self-consistency
        # sc_cmd = cot_cmd + " --n-completion 40"
        # print(sc_cmd)
        # os.system(sc_cmd)


