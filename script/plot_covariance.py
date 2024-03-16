"""This script plots the covariance of ndt.
"""

import argparse
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_csv", type=pathlib.Path)
    return parser.parse_args()


def plot_ellipse(mean, cov, color, label, scale):
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(eigenvalues) * scale
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        color=color,
        alpha=0.5,
        fill=False,
        label=f"{label}(scale={scale})",
    )
    plt.gca().add_patch(ellipse)


if __name__ == "__main__":
    args = parse_args()
    result_csv = args.result_csv

    df_result = pd.read_csv(result_csv)

    """
    df_result
    index,score,x,y,elapsed_la,cov_by_la_00,cov_by_la_01,cov_by_la_10,cov_by_la_11,elapsed_mndt,cov_by_mndt_00,cov_by_mndt_01,cov_by_mndt_10,cov_by_mndt_11,elapsed_mndt_score,cov_by_mndt_score_00,cov_by_mndt_score_01,cov_by_mndt_score_10,cov_by_mndt_score_11
    0,3.429885,81377.359375,49916.902344,0.000000,0.000110,0.000007,0.000007,0.000116,112.322000,0.000017,0.000000,0.000000,0.000003,12.878000,0.002066,0.000650,0.000650,0.002814
    1,3.412068,81377.359375,49916.902344,0.000000,0.000111,0.000007,0.000007,0.000117,62.720000,0.187296,0.104792,0.104792,0.226846,12.767000,0.002203,0.000831,0.000831,0.003018
    2,3.424125,81377.359375,49916.902344,0.000000,0.000109,0.000007,0.000007,0.000116,61.792000,0.161584,0.092505,0.092505,0.237047,12.803000,0.002010,0.000674,0.000674,0.002697
    ...
    """

    # plot time
    plt.figure()
    plt.plot(df_result["elapsed_la"], label="Laplace Approximation")
    plt.plot(df_result["elapsed_mndt"], label="Multi NDT")
    plt.plot(df_result["elapsed_mndt_score"], label="Multi NDT Score")
    plt.xlabel("Frame")
    plt.ylabel("Time [msec]")
    plt.legend()
    plt.tight_layout()
    save_path = result_csv.parent / "time.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {save_path}")
    plt.close()

    cov_by_la = df_result[
        ["cov_by_la_00", "cov_by_la_01", "cov_by_la_10", "cov_by_la_11"]
    ].values
    cov_by_mndt = df_result[
        ["cov_by_mndt_00", "cov_by_mndt_01", "cov_by_mndt_10", "cov_by_mndt_11"]
    ].values

    for i in range(2):
        for j in range(2):
            plt.subplot(2, 2, i * 2 + j + 1)
            plt.plot(cov_by_la[:, i * 2 + j], label="Laplace Approximation")
            plt.plot(cov_by_mndt[:, i * 2 + j], label="Multi NDT")
            plt.ylabel(f"cov_{i}{j}")
            plt.legend()
    plt.tight_layout()
    save_path = result_csv.parent / "covariance.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {save_path}")
    plt.close()

    mean_x = df_result["initial_x"].mean()
    mean_y = df_result["initial_y"].mean()

    df_result["initial_x"] -= mean_x
    df_result["initial_y"] -= mean_y

    cov_default = 0.0225 * np.eye(2)

    # plot each frame
    output_dir = result_csv.parent / "covariance_each_frame"
    output_dir.mkdir(exist_ok=True, parents=True)
    progress = tqdm(total=len(df_result))
    for i, row in df_result.iterrows():
        progress.update(1)
        # plot ellipse
        plt.figure()
        cov_by_la = row[
            ["cov_by_la_00", "cov_by_la_01", "cov_by_la_10", "cov_by_la_11"]
        ].values.reshape(2, 2)
        cov_by_mndt = row[
            ["cov_by_mndt_00", "cov_by_mndt_01", "cov_by_mndt_10", "cov_by_mndt_11"]
        ].values.reshape(2, 2)
        cov_by_mndt_score = row[
            [
                "cov_by_mndt_score_00",
                "cov_by_mndt_score_01",
                "cov_by_mndt_score_10",
                "cov_by_mndt_score_11",
            ]
        ].values.reshape(2, 2)
        x, y = row["initial_x"], row["initial_y"]
        plot_ellipse([x, y], cov_default, "green", "Default", 10)
        plot_ellipse([x, y], cov_by_la, "blue", "Laplace Approximation", 100)
        plot_ellipse([x, y], cov_by_mndt, "red", "Multi NDT", 10)
        plot_ellipse([x, y], cov_by_mndt_score, "orange", "Multi NDT Score", 10)
        plt.scatter(
            df_result["initial_x"][0:i], df_result["initial_y"][0:i], color="black", s=1
        )
        plt.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0))
        plt.grid()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.xlim(x - 15, x + 15)
        plt.ylim(y - 15, y + 15)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.savefig(output_dir / f"{i:08d}.png", bbox_inches="tight", pad_inches=0.05)
        plt.close()
