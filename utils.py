import numpy as np
import pandas as pd

def append_results(results, cur_result):
    for key in results:
        if key in cur_result:
            results[key].append(cur_result[key])
        else:
            results[key].append(np.nan)

    return results


def append_history(history, example, lf):
    for key in ["sentence", "idx", "label"]:
        if key in history:
            history[key].append(example[key])
        else:
            history[key] = [example[key]]

    if lf is None:
        lf_info = "None"
    else:
        lf_info = lf.info()

    if "lf" in history:
        history["lf"].append(lf_info)
    else:
        history["lf"] = [lf_info]

    return history


def save_results(results, save_path):
    df = pd.DataFrame(results)
    df.to_csv(save_path)