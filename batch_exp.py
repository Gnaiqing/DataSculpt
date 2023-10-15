import os

### Constants
data_root = "./data/wrench_data"
relation_extraction_datasets = ("chemprot", "cdr", "spouse", "semeval")
imbalanced_datasets = ("sms", "cdr", "spouse", "arxiv_abstract")
multiclass_datasets = ("chemprot", "agnews", "trec", "medical_abstract")

### Modify the configurations below
datasets = [
    "youtube",
    "sms",
    "imdb",
    "yelp",
    # "agnews",
    # "trec",
    # "medical_abstract",
    # "arxiv_abstract",
    # "chemprot",
    # "cdr",
    # "spouse",
    # "semeval"
]

samplers = [
    "passive",
    # "uncertain",
    # "SEU",
    # "weighted"
]

acc = 0.6
agent = "chatgpt"
feature_extractor = "bert"
evaluate_base_prompt = False  # base prompt
evaluate_cot_prompt = False  # chain of thought
evaluate_sc_prompt = True   # self-consistency
example_selection = "random"  # "random" or "neighbor"

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
              f"--example-selection {example_selection} --label-model Snorkel --display --save-wandb"
        if dataset in ["cdr", "spouse"]:
            # add default class for non-covered data
            cmd += " --default-class 0"

        if evaluate_base_prompt:
            print(cmd)
            os.system(cmd)

        # chain of thought
        cot_cmd = cmd + " --return-explanation"
        if evaluate_cot_prompt:
            print(cot_cmd)
            os.system(cot_cmd)

        # self-consistency
        sc_cmd = cot_cmd + " --n-completion 10"
        if evaluate_sc_prompt:
            print(sc_cmd)
            os.system(sc_cmd)


