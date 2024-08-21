import argparse
from data_utils import load_wrench_data
from label_model import get_wrench_label_model
from end_model import train_disc_model
from wrench.search import grid_search
from wrench.search_space import SEARCH_SPACE
from utils import print_dataset_stats, evaluate_lfs, evaluate_labels,evaluate_disc_model
import numpy as np
import wandb
import json
import os
from openai import OpenAI
from lf_agent import completion_with_backoff


def prompt_language_model(client, prompt_template, instance, model):
    """
    Prompt language model
    """
    prompt = prompt_template.replace("[TEXT]", instance["text"])
    if "entity1" in instance:
        prompt = prompt.replace("[ENTITY1]", instance["entity1"])
    if "entity2" in instance:
        prompt = prompt.replace("[ENTITY2]", instance["entity2"])

    messages = [
        {"role": "user", "content": prompt}
    ]
    response = completion_with_backoff(
        client,
        model=model,
        messages=messages
    )
    return prompt, response



def main(args):
    train_dataset, valid_dataset, test_dataset = load_wrench_data(data_root=args.dataset_path,
                                                                  dataset_name=args.dataset_name,
                                                                  feature=args.feature_extractor,
                                                                  revert_index=False)
    print(f"Dataset path: {args.dataset_path}, name: {args.dataset_name}")
    if args.sample_frac < 1:
        train_dataset = train_dataset.sample(alpha=args.sample_frac)
        valid_dataset = valid_dataset.sample(alpha=args.sample_frac)
    print_dataset_stats(train_dataset, split="train")
    print_dataset_stats(valid_dataset, split="valid")
    print_dataset_stats(test_dataset, split="test")

    client = OpenAI(api_key=args.api_key)

    if args.save_wandb:
        group_id = wandb.util.generate_id()
        config_dict = vars(args)
        config_dict["method"] = "PromptedLF"
        config_dict["group_id"] = group_id
        wandb.init(
            project="DataSculpt",
            config=config_dict
        )

    # load prompt templates
    with open(args.prompt_path, "r") as f:
        prompt_templates = json.load(f)
        dataset_prompts = prompt_templates[args.dataset_name]

    L_train = []
    L_valid = []

    prompt_token_usage = 0
    completion_token_usage = 0

    if args.load_lf_path is not None:
        weak_labels = np.load(args.load_lf_path)
        L_train = weak_labels["L_train"]
        L_valid = weak_labels["L_valid"]
        train_dataset.weak_labels = L_train
        valid_dataset.weak_labels = L_valid
    else:
        if args.dataset_name == "sms":
            # transform the templates
            pos_keywords = [words.strip() for words in dataset_prompts["positive_keywords"].split(",")]
            neg_keywords = [words.strip() for words in dataset_prompts["negative_keywords"].split(",")]
            prompt_template = dataset_prompts["prompt"]
            dataset_prompts = []
            for pos_keyword in pos_keywords:
                dataset_prompts.append({"prompt": prompt_template.replace("[KEYWORDS]", pos_keyword), "pos_label": 1})

            for neg_keyword in neg_keywords:
                dataset_prompts.append({"prompt": prompt_template.replace("[KEYWORDS]", neg_keyword), "pos_label": 0})

        for dataset_type in ["train", "valid"]:
            if dataset_type == "train":
                dataset = train_dataset
            else:
                dataset = valid_dataset

            for template_dict in dataset_prompts:
                weak_labels = []
                prompt_template = template_dict["prompt"]
                for i in range(len(dataset)):
                    instance = dataset.examples[i]
                    prompt, response = prompt_language_model(client, prompt_template, instance, args.prompt_model)
                    response_text = response.choices[0].message.content
                    prompt_token_usage += response.usage.prompt_tokens
                    completion_token_usage += response.usage.completion_tokens
                    if args.display:
                        print("Prompt: ", prompt)
                        print("Response: ", response_text)

                    if "yes" in response_text.lower():
                        weak_labels.append(template_dict["pos_label"])
                    elif "neg_label" in template_dict and "no" in response_text.lower():
                        weak_labels.append(template_dict["neg_label"])
                    else:
                        # abstain on the instance
                        weak_labels.append(-1)

                if dataset_type == "train":
                    L_train.append(weak_labels)
                else:
                    L_valid.append(weak_labels)

        L_train = np.array(L_train, dtype=int).T
        L_valid = np.array(L_valid, dtype=int).T
        train_dataset.weak_labels = L_train
        valid_dataset.weak_labels = L_valid
        if args.save_lf_path is not None:
            os.makedirs(os.path.dirname(args.save_lf_path), exist_ok=True)
            np.savez(args.save_lf_path, L_train=L_train, L_valid=L_valid)

    label_model = get_wrench_label_model(args.label_model, verbose=False)
    search_space = SEARCH_SPACE.get(args.label_model, None)
    if args.tune_metric == "f1":
        metric = "f1_binary" if train_dataset.n_class == 2 else "f1_macro"
    else:
        metric = "acc"

    if search_space is not None and args.tune_label_model:
        searched_paras = grid_search(label_model, dataset_train=train_dataset,
                                     dataset_valid=valid_dataset,
                                     metric=metric, direction='auto',
                                     search_space=search_space,
                                     n_repeats=1, n_trials=100, parallel=False)
        label_model = get_wrench_label_model(args.label_model, **searched_paras, verbose=False)

    label_model.fit(dataset_train=train_dataset,
                    dataset_valid=valid_dataset,
                    )
    ys_tr = label_model.predict(L_train)
    ys_tr_soft = label_model.predict_proba(L_train)
    # train_covered_indices = (np.max(L_train, axis=1) != -1) & (ys_tr != -1)  # indices covered by LFs
    train_covered_indices = np.max(L_train, axis=1) != -1
    ys_tr[~train_covered_indices] = -1
    gt_train_labels = np.array(train_dataset.labels)
    lf_train_stats = evaluate_lfs(gt_train_labels, L_train, n_class=train_dataset.n_class)
    train_label_stats = evaluate_labels(gt_train_labels, ys_tr, n_class=train_dataset.n_class)
    # train end model
    xs_tr = train_dataset.features[train_covered_indices, :]
    ys_tr_soft = ys_tr_soft[train_covered_indices, :]
    ys_tr_hard = ys_tr[train_covered_indices]
    disc_model = train_disc_model(model_type=args.end_model,
                                  xs_tr=xs_tr,
                                  ys_tr_soft=ys_tr_soft,
                                  ys_tr_hard=ys_tr_hard,
                                  valid_dataset=valid_dataset,
                                  soft_training=args.use_soft_labels,
                                  tune_end_model=args.tune_end_model,
                                  tune_metric=args.tune_metric,
                                  seed=0)
    test_perf = evaluate_disc_model(disc_model, test_dataset)
    print(f"Test performance: {test_perf}")

    if args.save_wandb:
        wandb.run.summary["lf_num"] = L_train.shape[1]
        wandb.run.summary["total_token_usage"] = prompt_token_usage + completion_token_usage
        wandb.run.summary["prompt_token_usage"] = prompt_token_usage
        wandb.run.summary["completion_token_usage"] = completion_token_usage
        wandb.run.summary["lf_acc_avg"] = lf_train_stats["lf_acc_avg"]
        wandb.run.summary["lf_cov_avg"] = lf_train_stats["lf_cov_avg"]
        wandb.run.summary["lf_overlap_avg"] = lf_train_stats["lf_overlap_avg"]
        wandb.run.summary["lf_conflict_avg"] = lf_train_stats["lf_conflict_avg"]
        wandb.run.summary["train_precision"] = train_label_stats["accuracy"]
        wandb.run.summary["train_coverage"] = train_label_stats["coverage"]
        wandb.run.summary["test_acc"] = test_perf["acc"]
        wandb.run.summary["test_f1"] = test_perf["f1"]
        wandb.run.summary["test_auc"] = test_perf["auc"]
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--dataset-path", type=str, default="./data/wrench_data", help="dataset path")
    parser.add_argument("--dataset-name", type=str, default="youtube", help="dataset name")
    parser.add_argument("--feature-extractor", type=str, default="bert", help="feature for training end model")
    parser.add_argument("--sample-frac", type=float, default=1.0)
    # models
    parser.add_argument("--api-key", type=str, help="openai api key")
    parser.add_argument("--prompt-model", type=str, default="gpt-3.5-turbo", help="prompt model")
    parser.add_argument("--label-model", type=str, default="Snorkel", choices=["Snorkel", "MeTaL", "MV"],
                        help="label model used in DP paradigm")
    parser.add_argument("--use-soft-labels", action="store_true",
                        help="set to true if use soft labels when training end model")
    parser.add_argument("--end-model", type=str, default="logistic", choices=["logistic", "mlp"],
                        help="end model in DP paradigm")
    parser.add_argument("--tune-label-model", action="store_true", help="tune label model hyperparameters")
    parser.add_argument("--tune-end-model", action="store_true", help="tune end model hyperparameters")
    parser.add_argument("--tune-metric", type=str, default="acc",
                        help="evaluation metric used to tune model hyperparameters")
    # experiment
    parser.add_argument("--prompt-path", type=str, default="./wrench_prompts.json", help="path to save prompts")
    parser.add_argument("--save-wandb", action="store_true", help="save experiment results to wandb")
    parser.add_argument("--save-lf-path", type=str, default=None, help="save generated LFs")
    parser.add_argument("--load-lf-path", type=str, default=None, help="load generated LFs")
    parser.add_argument("--display", action="store_true", help="display queries and responses")
    args = parser.parse_args()
    main(args)