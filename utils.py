import numpy as np
import pandas as pd


def append_results(results, cur_result):
    for key in results:
        if key in cur_result:
            results[key].append(cur_result[key])
        else:
            results[key].append(np.nan)

    return results


def save_results(results, save_path):
    df = pd.DataFrame(results)
    df.to_csv(save_path)


def print_dataset_stats(dataset, split="train"):
    print("{} size: {}".format(split, len(dataset)))
    values, counts = np.unique(dataset.labels, return_counts=True)
    freq = counts / len(dataset)
    print("Label distribution: ", freq)


