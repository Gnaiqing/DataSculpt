data_path=$1
model=$2
# sms Yelp AmazonReview
# 0.7 0.8s
# --save-wandb
for data_name in youtube sms Yelp AmazonReview; do
  for model_size in 13b; do
      model_path=$model-$model_size
      echo $model_path
       for acc in 0.6; do
          python main.py --dataset-path $data_path --dataset-name $data_name --lf-acc-threshold $acc --lf-agent $model --lf-llm-model $model_path --use-soft-labels --save-wandb
      done
  done
done

#    parser.add_argument("--dataset-path", type=str, default="glue", help="dataset path (or benchmark name)")
#     parser.add_argument("--dataset-name", type=str, default="sst2", help="dataset name")
#     parser.add_argument("--feature-extractor", type=str, default="tfidf", help="feature for training end model")
#     # sampler
#     parser.add_argument("--sampler", type=str, default="passive", help="sample selector")
#     # data programming
#     parser.add_argument("--label-model", type=str, default="snorkel", help="label model used in DP paradigm")
#     parser.add_argument("--use-soft-labels", action="store_true", help="set to true if use soft labels when training end model")
#     parser.add_argument("--end-model", type=str, default="logistic", help="end model in DP paradigm")
#     # label function
#     parser.add_argument("--lf-agent", type=str, default="simulated", help="agent that return candidate LFs")
#     parser.add_argument("--lf-type", type=str, default="keyword", help="LF family")
#     parser.add_argument("--lf-filter", type=str, nargs="+", default=["acc", "unique"], help="filters for simulated agent")
#     parser.add_argument("--lf-acc-threshold", type=float, default=0.5, help="LF accuracy threshold for simulated agent")
#     parser.add_argument("--lf-llm-model", type=str, default="gpt-3.5-turbo")
#     # experiment
#     parser.add_argument("--num-query", type=int, default=50, help="total selected samples")
#     parser.add_argument("--train-iter", type=int, default=5, help="evaluation interval")
#     parser.add_argument("--results-dir", type=str,default="./results")
#     parser.add_argument("--runs", type=int, default=5)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--display", action="store_true")
#     parser.add_argument("--save-wandb", action="store_true")
#     parser.add_argument("--tag", type=str, default="0")