"""This script compares two results of a regression test.
"""

import argparse
import pathlib
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("current_output_dir", type=pathlib.Path)
    parser.add_argument("reference_output_dir", type=pathlib.Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    current_output_dir = args.current_output_dir
    reference_output_dir = args.reference_output_dir

    current_df = pd.read_csv(current_output_dir / "result.csv")
    reference_df = pd.read_csv(reference_output_dir / "result.csv")

    """
    elapsed_milliseconds,nvtl_score,tp_score
    574.228000,3.245870,6.492360
    7.980000,3.222864,6.484424
    8.261000,3.249374,6.525939
    8.245000,3.248322,6.554400
    ...
    """

    # The first data point in results is slow, so remove it.
    current_df = current_df.iloc[1:]
    reference_df = reference_df.iloc[1:]

    # calculate the ratio of elapsed time
    elapsed_milliseconds_ratio = (
        current_df["elapsed_milliseconds"] / reference_df["elapsed_milliseconds"]
    )
    elapsed_milliseconds_ratio_mean = elapsed_milliseconds_ratio.mean()
    print(f"{elapsed_milliseconds_ratio_mean=:.3f} (current / reference)")
    assert elapsed_milliseconds_ratio_mean < 1.1, "The elapsed time is too slow."

    # plot
    plt.plot(current_df["elapsed_milliseconds"], label="current")
    plt.plot(reference_df["elapsed_milliseconds"], label="reference")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("elapsed_milliseconds")
    plt.grid()
    plt.savefig(
        current_output_dir / "elapsed_milliseconds.png",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close()

    # calculate difference of score
    score_diff_nvtl = current_df["nvtl_score"] - reference_df["nvtl_score"]
    all_zero_nvtl = (score_diff_nvtl == 0).all()

    score_diff_tp = current_df["tp_score"] - reference_df["tp_score"]
    all_zero_tp = (score_diff_tp == 0).all()

    if all_zero_nvtl and all_zero_tp:
        print("The scores are perfectly the same.")
        print("OK")
        exit(0)

    score_diff_abs_mean_nvtl = score_diff_nvtl.abs().mean()
    score_diff_abs_mean_tp = score_diff_tp.abs().mean()
    print(f"{score_diff_abs_mean_nvtl=:.3f}, {score_diff_abs_mean_tp=:.3f}")
    assert score_diff_abs_mean_nvtl < 0.1, "The nvtl_score is too different."
    assert score_diff_abs_mean_tp < 0.1, "The tp_score is too different."
