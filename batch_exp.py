import os

### Constants
data_root = "/root/datasets"
relation_extraction_datasets = ("spouse",)
imbalanced_datasets = ("sms", "spouse")
multiclass_datasets = ("agnews",)

### Modify the configurations below
datasets = [
    "youtube",
    "sms",
    "imdb",
    "yelp",
    "agnews",
    "spouse",
]

samplers = [
    "passive",
    # "uncertain",
    # "SEU",
]

filters = [
    "'acc' 'overlap'",
    # "acc",
    # "overlap"
]

acc = 0.6
agent = "chatgpt"
feature_extractor = "bert"
evaluate_base_prompt = False  # base prompt
evaluate_cot_prompt = False  # chain of thought
evaluate_sc_prompt = True  # self-consistency
evaluate_kate_prompt = False # kate prompt
example_selection = "random"  # "random" or "neighbor"
llm = "gpt-3.5-turbo"  # gpt-3.5-turbo, gpt-4-0613
api_key = ""  # openai api key

for dataset in datasets:
    for sampler in samplers:
        for lf_filter in filters:
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
                  f" --lf-filter {lf_filter} --lf-llm-model {llm} --label-model Snorkel --display --save-wandb --api-key {api_key}"
            if llm == "gpt-4-0613":
                cmd += " --sleep-time 10"

            if dataset == "spouse":
                # add default class for non-covered data
                cmd += " --default-class 0"

            if evaluate_base_prompt:
                print(cmd + " --save-lf")
                os.system(cmd + " --save-lf")

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

            # kate prompt
            kate_cmd = sc_cmd + " --example-selection neighbor"
            if evaluate_kate_prompt:
                print(kate_cmd)
                os.system(kate_cmd)


