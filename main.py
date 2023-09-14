import argparse
from data_utils import load_wrench_data, relation_extraction_datasets
from sampler import get_sampler
from lf_agent import get_lf_agent
from lf_family import create_label_matrix
from label_model import get_label_model, Snorkel
from end_model import train_disc_model
from utils import print_dataset_stats, evaluate_lfs, evaluate_labels, evaluate_disc_model
import numpy as np
from sklearn.metrics import accuracy_score
import wandb
import pprint


def main(args):
    train_dataset, valid_dataset, test_dataset = load_wrench_data(args.dataset_path, args.dataset_name, args.feature_extractor)
    print(f"Dataset path: {args.dataset_path}, name: {args.dataset_name}")
    print_dataset_stats(train_dataset, split="train")
    print_dataset_stats(valid_dataset, split="valid")
    print_dataset_stats(test_dataset, split="test")

    if args.save_wandb:
        group_id = wandb.util.generate_id()
        config_dict = vars(args)
        config_dict["method"] = "LLMDP"
        config_dict["group_id"] = group_id

    rng = np.random.default_rng(args.seed)
    for run in range(args.runs):
        if args.save_wandb:
            wandb.init(
                project="LLMDP",
                config=config_dict
            )
            wandb.define_metric("test_acc", summary="mean")
            wandb.define_metric("test_auc", summary="mean")
            wandb.define_metric("test_f1", summary="mean")
            wandb.define_metric("train_precision", summary="mean")
            wandb.define_metric("train_coverage", summary="mean")

        seed = rng.choice(10000)
        sampler = get_sampler(train_dataset=train_dataset,
                              sampler_type=args.sampler,
                              seed=seed
                              )
        lf_agent = get_lf_agent(train_dataset=train_dataset,
                                valid_dataset=valid_dataset,
                                agent_type=args.lf_agent,
                                lf_type=args.lf_type,
                                filter_methods=args.lf_filter,
                                acc_threshold=args.lf_acc_threshold,
                                model=args.lf_llm_model,
                                seed=seed,
                                dataset_name=args.dataset_name,
                                # following configs are for GPT/llama models
                                example_per_class=args.example_per_class,
                                example_selection=args.example_selection,
                                return_explanation=args.return_explanation,
                                play_expert_role=args.play_expert_role,
                                dp_aware=args.dp_aware,
                                )
        al_model = None
        label_model = None
        lfs = []
        gt_labels = []
        response_labels = []

        for t in range(args.num_query):
            query_idx = sampler.sample(al_model=al_model)[0]
            if args.display:
                if args.dataset_name in relation_extraction_datasets:
                    text = train_dataset.examples[query_idx]["text"]
                    entity1 = train_dataset.examples[query_idx]["entity1"]
                    entity2 = train_dataset.examples[query_idx]["entity2"]
                    print("Query [{}]: {} <{}> <{}>".format(query_idx, text, entity1, entity2))
                else:
                    print("Query [{}]: {}".format(query_idx,train_dataset.examples[query_idx]["text"]))
            lf = lf_agent.create_lf(query_idx)
            if args.display:
                print("LF: ", lf.info())
                print("Ground truth: ", train_dataset.labels[query_idx])

            if lf.keyword != "NA":
                gt_labels.append(train_dataset.labels[query_idx])
                response_labels.append(lf.label)
                lfs.append(lf)
                lf_labels = [lf.label for lf in lfs]
                L_train = create_label_matrix(train_dataset, lfs)
                L_val = create_label_matrix(valid_dataset, lfs)
                if np.min(lf_labels) != np.max(lf_labels):
                    if len(lfs) > 3:
                        label_model = get_label_model(args.label_model, train_dataset.n_class)
                    else:
                        label_model = get_label_model("mv", train_dataset.n_class)

                    if isinstance(label_model, Snorkel):
                        label_model.fit(L_train, L_val, np.array(valid_dataset.labels), tune_label_model=args.tune_label_model)

            else:
                gt_labels.append(train_dataset.labels[query_idx])
                if lf.label is not None:
                    response_labels.append(lf.label)
                else:
                    response_labels.append(-1)

            if t % args.train_iter == args.train_iter - 1:
                if label_model is not None:
                    # evaluate LF quality
                    gt_train_labels = np.array(train_dataset.labels)
                    lf_train_stats = evaluate_lfs(gt_train_labels, L_train, np.array(lf_labels), n_class=train_dataset.n_class)
                    gt_valid_labels = np.array(valid_dataset.labels)
                    lf_val_stats = evaluate_lfs(gt_valid_labels, L_val, np.array(lf_labels), n_class=valid_dataset.n_class)

                    # get label model predictions
                    ys_tr = label_model.predict(L_train)
                    ys_tr_soft = label_model.predict_proba(L_train)
                    train_covered_indices = (np.max(L_train, axis=1) != -1) & (ys_tr != -1)  # indices covered by LFs
                    ys_tr[~train_covered_indices] = -1

                    ys_val = label_model.predict(L_val)
                    valid_covered_indices = (np.max(L_val, axis=1) != -1) & (ys_val != -1)  # indices covered by LFs
                    ys_val[~valid_covered_indices] = -1

                    # evaluate label quality
                    response_acc = accuracy_score(gt_labels, response_labels)
                    train_label_stats = evaluate_labels(gt_train_labels, ys_tr, n_class=train_dataset.n_class)
                    valid_label_stats = evaluate_labels(gt_valid_labels, ys_val, n_class=valid_dataset.n_class)

                    # train end model
                    xs_tr = train_dataset.features[train_covered_indices, :]
                    xs_u = train_dataset.features[~train_covered_indices, :]
                    ys_tr = ys_tr[train_covered_indices]
                    ys_tr_soft = ys_tr_soft[train_covered_indices, :]
                    if np.min(ys_tr) != np.max(ys_tr):
                        disc_model = train_disc_model(model_type=args.end_model,
                                                      xs_tr=xs_tr,
                                                      ys_tr_soft=ys_tr_soft,
                                                      ys_tr_hard=ys_tr,
                                                      xs_u=xs_u,
                                                      valid_dataset=valid_dataset,
                                                      soft_training=args.use_soft_labels,
                                                      ssl_method=args.ssl_method,
                                                      tune_end_model=args.tune_end_model,
                                                      tune_metric=args.tune_metric,
                                                      seed=seed)
                        # evaluate end model performance
                        test_preds = disc_model.predict(test_dataset.features)
                        gt_test_labels = np.array(test_dataset.labels)
                        test_stats = evaluate_labels(gt_test_labels, test_preds, n_class=test_dataset.n_class)
                        test_perf = evaluate_disc_model(disc_model, test_dataset)

                        valid_preds = disc_model.predict(valid_dataset.features)
                        valid_stats = evaluate_labels(gt_valid_labels, valid_preds, n_class=valid_dataset.n_class)

                    else:
                        test_perf = {"acc": np.nan, "f1": np.nan, "auc": np.nan}
                        valid_stats = {}
                        test_stats = {}

                    cur_result = {
                        "num_query": t+1,
                        "lf_num": len(lfs),
                        "lf_acc_avg": lf_train_stats["lf_acc_avg"],
                        "lf_cov_avg": lf_train_stats["lf_cov_avg"],
                        "response_acc": response_acc,
                        "train_precision": train_label_stats["accuracy"],
                        "train_coverage": train_label_stats["coverage"],
                        "test_acc": test_perf["acc"],
                        "test_f1": test_perf["f1"],
                        "test_auc": test_perf["auc"]
                    }
                    if args.save_wandb:
                        wandb.log(cur_result)

                    if args.display:
                        print(f"After {t+1} iterations:")
                        print("Train LF stats:")
                        pprint.pprint(lf_train_stats)
                        print("Valid LF stats:")
                        pprint.pprint(lf_val_stats)
                        print("Train label stats (label model output):")
                        pprint.pprint(train_label_stats)
                        print("Valid label stats (label model output):")
                        pprint.pprint(valid_label_stats)
                        print("Valid prediction stats (end model output):")
                        pprint.pprint(valid_stats)
                        print("Test prediction stats (end model output):")
                        pprint.pprint(test_stats)
                else:
                    # evaluate LF quality
                    gt_train_labels = np.array(train_dataset.labels)
                    lf_train_stats = evaluate_lfs(gt_train_labels, L_train, np.array(lf_labels),
                                            n_class=train_dataset.n_class)
                    gt_valid_labels = np.array(valid_dataset.labels)
                    lf_val_stats = evaluate_lfs(gt_valid_labels, L_val, np.array(lf_labels),
                                                n_class=valid_dataset.n_class)
                    response_acc = accuracy_score(gt_labels, response_labels)
                    cur_result = {
                        "num_query": t + 1,
                        "lf_num": len(lfs),
                        "lf_acc_avg": lf_train_stats["lf_acc_avg"],
                        "lf_cov_avg": lf_train_stats["lf_cov_avg"],
                        "response_acc": response_acc,
                        "train_precision": np.nan,
                        "train_coverage": np.nan,
                        "test_acc": np.nan,
                        "test_f1": np.nan,
                        "test_auc": np.nan
                    }
                    if args.save_wandb:
                        wandb.log(cur_result)
                    if args.display:
                        print(f"After {t+1} iterations:")
                        print("Train LF stats:")
                        pprint.pprint(lf_train_stats)
                        print("Valid LF stats:")
                        pprint.pprint(lf_val_stats)
                        print("No enough LFs. Skip training label model and end model.")

        if args.save_wandb:
            wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--dataset-source", type=str, default="wrench", choices=["wrench", "nemo", "hub"])
    parser.add_argument("--dataset-path", type=str, default="./data/wrench_data", help="dataset path")
    parser.add_argument("--dataset-name", type=str, default="youtube", help="dataset name")
    parser.add_argument("--feature-extractor", type=str, default="tfidf", help="feature for training end model")
    # sampler
    parser.add_argument("--sampler", type=str, default="passive", help="sample selector")
    # data programming
    parser.add_argument("--label-model", type=str, default="snorkel", help="label model used in DP paradigm")
    parser.add_argument("--use-soft-labels", action="store_true", help="set to true if use soft labels when training end model")
    parser.add_argument("--end-model", type=str, default="logistic", help="end model in DP paradigm")
    parser.add_argument("--ssl-method", type=str, default=None, choices=[None, "self-training"])
    parser.add_argument("--tune-label-model", type=bool, default=True, help="tune label model hyperparameters")
    parser.add_argument("--tune-end-model", type=bool, default=True, help="tune end model hyperparameters")
    parser.add_argument("--tune-metric", type=str, default="acc")
    # label function
    parser.add_argument("--lf-agent", type=str, default="simulated", help="agent that return candidate LFs")
    parser.add_argument("--lf-type", type=str, default="keyword", help="LF family")
    parser.add_argument("--lf-filter", type=str, nargs="+", default=["acc", "unique"], help="filters for simulated agent")
    parser.add_argument("--lf-acc-threshold", type=float, default=0.5, help="LF accuracy threshold for simulated agent")
    # prompting method
    parser.add_argument("--lf-llm-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--example-per-class", type=int, default=1)
    parser.add_argument("--return-explanation", action="store_true")
    parser.add_argument("--play-expert-role", action="store_true")
    parser.add_argument("--dp-aware", action="store_true")
    parser.add_argument("--example-selection", type=str, default="random")
    # experiment
    parser.add_argument("--num-query", type=int, default=50, help="total selected samples")
    parser.add_argument("--train-iter", type=int, default=5, help="evaluation interval")
    parser.add_argument("--results-dir", type=str,default="./results")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save-wandb", action="store_true")
    parser.add_argument("--tag", type=str, default="0")
    args = parser.parse_args()
    main(args)


