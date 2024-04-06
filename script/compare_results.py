"""This script compares two results of a regression test.
"""

import argparse
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("kinematic_state_csv", type=pathlib.Path)
    parser.add_argument("result_csv", type=pathlib.Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    kinematic_state_csv = args.kinematic_state_csv
    result_csv = args.result_csv

    df_gt = pd.read_csv(kinematic_state_csv)
    df_result = pd.read_csv(result_csv)

    """
    timestamp       pose_x        pose_y     pose_z    quat_w    quat_x    quat_y    quat_z
    0   41.18334  81377.34375  49916.894531  41.205414  0.953717  0.000178 -0.007339  0.300616
    1   41.28334  81377.34375  49916.894531  41.205414  0.953717  0.000178 -0.007339  0.300616
    2   41.38334  81377.34375  49916.894531  41.205414  0.953717  0.000178 -0.007339  0.300615
    3   41.48334  81377.34375  49916.894531  41.205414  0.953717  0.000178 -0.007339  0.300617
    4   41.58334  81377.34375  49916.894531  41.205414  0.953717  0.000178 -0.007339  0.300616
    ndt_omp_elapsed_msec  ndt_omp_pose00  ndt_omp_pose01  ndt_omp_pose02  ndt_omp_pose03  ...  fast_gicp_pose23  fast_gicp_pose30  fast_gicp_pose31  fast_gicp_pose32  fast_gicp_pose33
    0                12.588        0.819315       -0.573181       -0.013656    81377.359375  ...         41.226273               0.0               0.0               0.0               1.0
    1                 1.251        0.819355       -0.573131       -0.013373    81377.359375  ...         41.232796               0.0               0.0               0.0               1.0
    2                 1.277        0.819382       -0.573089       -0.013508    81377.367188  ...         41.234299               0.0               0.0               0.0               1.0
    3                 1.284        0.819396       -0.573067       -0.013605    81377.367188  ...         41.232098               0.0               0.0               0.0               1.0
    4                 1.249        0.819408       -0.573049       -0.013621    81377.359375  ...         41.230759               0.0               0.0               0.0               1.0

    [5 rows x 34 columns]
    """

    methods = ["ndt_omp", "fast_gicp"]

    # plot elapsed_msec
    for method in methods:
        plt.plot(df_result[f"{method}_elapsed_msec"][1:], label=method)

    plt.legend()
    plt.grid()
    plt.xlabel("frame number")
    plt.ylabel("elapsed [msec]")

    save_path = result_csv.parent / "elapsed_msec.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved to {save_path}")
    plt.close()

    # plot diff xyz
    gt_x = df_gt["pose_x"]
    gt_y = df_gt["pose_y"]
    gt_z = df_gt["pose_z"]
    fig, axs = plt.subplots(2, 2)
    for method in methods:
        pred_x = df_result[f"{method}_pose03"]
        pred_y = df_result[f"{method}_pose13"]
        pred_z = df_result[f"{method}_pose23"]
        diff_x = pred_x - gt_x
        diff_y = pred_y - gt_y
        diff_z = pred_z - gt_z
        diff = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
        axs[0, 0].plot(diff_x, label=method)
        axs[0, 0].set_ylabel("x[m]")
        axs[0, 0].grid(True)

        axs[0, 1].plot(diff_y)
        axs[0, 1].set_ylabel("y[m]")
        axs[0, 1].grid(True)

        axs[1, 0].plot(diff_z)
        axs[1, 0].set_ylabel("z[m]")
        axs[1, 0].grid(True)

        axs[1, 1].plot(diff)
        axs[1, 1].set_ylabel("norm[m]")
        axs[1, 1].grid(True)

    fig.tight_layout()
    fig.legend(loc="lower left", bbox_to_anchor=(0.4, 1.0))
    save_path = result_csv.parent / "diff_xyz.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved to {save_path}")
    plt.close()

    # plot roll, pitch, yaw
    gt_quat_w = df_gt["quat_w"]
    gt_quat_x = df_gt["quat_x"]
    gt_quat_y = df_gt["quat_y"]
    gt_quat_z = df_gt["quat_z"]
    gt_r = Rotation.from_quat(np.vstack([gt_quat_x, gt_quat_y, gt_quat_z, gt_quat_w]).T)
    fig, axs = plt.subplots(2, 2)
    for method in methods:
        rotate_matrix = np.zeros((len(df_result), 3, 3))
        rotate_matrix[:, 0, 0] = df_result[f"{method}_pose00"]
        rotate_matrix[:, 0, 1] = df_result[f"{method}_pose01"]
        rotate_matrix[:, 0, 2] = df_result[f"{method}_pose02"]
        rotate_matrix[:, 1, 0] = df_result[f"{method}_pose10"]
        rotate_matrix[:, 1, 1] = df_result[f"{method}_pose11"]
        rotate_matrix[:, 1, 2] = df_result[f"{method}_pose12"]
        rotate_matrix[:, 2, 0] = df_result[f"{method}_pose20"]
        rotate_matrix[:, 2, 1] = df_result[f"{method}_pose21"]
        rotate_matrix[:, 2, 2] = df_result[f"{method}_pose22"]
        pred_r = Rotation.from_matrix(rotate_matrix)
        diff_r = gt_r.inv() * pred_r
        diff_r_euler = diff_r.as_euler("xyz", degrees=True)
        axs[0, 0].plot(diff_r_euler[:, 0], label=method)
        axs[0, 0].set_ylabel("roll[deg]")
        axs[0, 0].grid(True)

        axs[0, 1].plot(diff_r_euler[:, 1])
        axs[0, 1].set_ylabel("pitch[deg]")
        axs[0, 1].grid(True)

        axs[1, 0].plot(diff_r_euler[:, 2])
        axs[1, 0].set_ylabel("yaw[deg]")
        axs[1, 0].grid(True)

        diff_r_norm = np.linalg.norm(diff_r.as_rotvec(), axis=1)
        diff_r_norm = np.rad2deg(diff_r_norm)
        axs[1, 1].plot(diff_r_norm)
        axs[1, 1].set_ylabel("norm[deg]")
        axs[1, 1].grid(True)

    fig.tight_layout()
    fig.legend(loc="lower left", bbox_to_anchor=(0.4, 1.0))
    save_path = result_csv.parent / "diff_rpy.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved to {save_path}")
    plt.close()
