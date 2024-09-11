"""This script converts a rosbag to regression test input data for NDT apps.
"""

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import argparse
import pathlib
import pandas as pd
from sensor_msgs_py import point_cloud2
import open3d as o3d
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("rosbag_path", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--pose_topic", type=str, default="/localization/kinematic_state")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rosbag_path = args.rosbag_path
    output_dir = args.output_dir
    pose_topic = args.pose_topic

    serialization_format = "cdr"
    storage_options = rosbag2_py.StorageOptions(
        uri=str(rosbag_path), storage_id="sqlite3"
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
    }

    target_topics = [
        pose_topic,
        "/localization/util/downsample/pointcloud",
    ]
    storage_filter = rosbag2_py.StorageFilter(topics=target_topics)
    reader.set_filter(storage_filter)

    kinematic_state_list = []
    pointcloud_list = []

    while reader.has_next():
        (topic, data, timestamp_rosbag) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        timestamp_header = (
            int(msg.header.stamp.sec) + int(msg.header.stamp.nanosec) * 1e-9
        )
        if topic == pose_topic:
            pose = msg.pose.pose
            twist = msg.twist.twist
            kinematic_state_list.append(
                {
                    "timestamp": timestamp_header,
                    "pose_x": pose.position.x,
                    "pose_y": pose.position.y,
                    "pose_z": pose.position.z,
                    "quat_w": pose.orientation.w,
                    "quat_x": pose.orientation.x,
                    "quat_y": pose.orientation.y,
                    "quat_z": pose.orientation.z,
                    "twist_linear_x": twist.linear.x,
                    "twist_linear_y": twist.linear.y,
                    "twist_linear_z": twist.linear.z,
                    "twist_angular_x": twist.angular.x,
                    "twist_angular_y": twist.angular.y,
                    "twist_angular_z": twist.angular.z,
                }
            )
        elif topic == "/localization/util/downsample/pointcloud":
            pointcloud_list.append(
                {
                    "timestamp": timestamp_header,
                    "pointcloud": msg,
                }
            )
        else:
            assert False, f"Unknown topic: {topic}"

    print(f"{len(kinematic_state_list)=}")
    print(f"{len(pointcloud_list)=}")

    df_kinematic_state = pd.DataFrame(kinematic_state_list)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sensor_pcd").mkdir(parents=True, exist_ok=True)

    index_kinematic_state = 0
    used_flag_kinematic_state = [False] * len(kinematic_state_list)

    for i, data in enumerate(pointcloud_list):
        timestamp = data["timestamp"]
        pointcloud = data["pointcloud"]  # sensor_msgs.msg.PointCloud2

        # find nearest kinematic_state
        while index_kinematic_state + 1 < len(kinematic_state_list):
            if kinematic_state_list[index_kinematic_state + 1]["timestamp"] > timestamp:
                break
            index_kinematic_state += 1

        if index_kinematic_state >= len(kinematic_state_list):
            break

        used_flag_kinematic_state[index_kinematic_state] = True
        index_kinematic_state += 1

        # convert pointcloud to pcd format
        pcd = point_cloud2.read_points(pointcloud, field_names=("x", "y", "z"))
        pcd = np.column_stack((pcd["x"], pcd["y"], pcd["z"]))
        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(pcd)

        # save pointcloud
        output_path = output_dir / "sensor_pcd" / f"pointcloud_{i:08d}.pcd"
        o3d.io.write_point_cloud(str(output_path), o3d_pointcloud)

    # save kinematic_state
    df_kinematic_state = df_kinematic_state[used_flag_kinematic_state]
    df_kinematic_state.reset_index(drop=True, inplace=True)
    df_kinematic_state.to_csv(
        output_dir / "kinematic_state.csv", index=False, float_format="%.6f"
    )
    print(f"Saved to {output_dir.resolve()}")
