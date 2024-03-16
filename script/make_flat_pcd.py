import argparse
import open3d as o3d
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pointcloud_map", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source_pointcloud_map = args.source_pointcloud_map
    pcd = o3d.io.read_point_cloud(str(source_pointcloud_map))
    print(pcd)

    bbox = pcd.get_axis_aligned_bounding_box()
    print(bbox)

    z_mean = np.mean(np.asarray(pcd.points)[:, 2])

    # Create a new point cloud with z=z_mean points
    # Add random noise to x and y to spread the points
    RANDOM_WIDTH = 50
    flat_pcd = o3d.geometry.PointCloud()
    flat_pcd.points = o3d.utility.Vector3dVector(
        [
            [
                p[0] + np.random.rand() * 2 * RANDOM_WIDTH - RANDOM_WIDTH,
                p[1] + np.random.rand() * 2 * RANDOM_WIDTH - RANDOM_WIDTH,
                z_mean,
            ]
            for p in pcd.points
        ]
    )

    # visualize
    # o3d.visualization.draw_geometries([flat_pcd])

    # save
    save_path = source_pointcloud_map.parent / "flat_pointcloud_map.pcd"
    o3d.io.write_point_cloud(
        str(save_path), flat_pcd, write_ascii=False, compressed=True
    )
    print(f"Saved: {save_path.absolute()}")
