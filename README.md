# ndt_omp
This package provides an OpenMP-boosted Normal Distributions Transform (and GICP) algorithm derived from pcl. The NDT algorithm is modified to be SSE-friendly and multi-threaded. It can run up to 10 times faster than its original version in pcl.

# multigrid ndt_omp
TIER IV has developed an extended version of `ndt_omp` for dynamic map loading functionality.
The difference from the `ndt_omp` is as follows:
- Instead of `setInputTarget` interface, `multigrid_ndt_omp` provides `addTarget` and `removeTarget` for more flexible target inputs.
- Only `RadiusSearch` is supported as a search method (`getNeighborhoodAtPointX` methods are disabled).

[![Build](https://github.com/koide3/ndt_omp/actions/workflows/build.yml/badge.svg)](https://github.com/koide3/ndt_omp/actions/workflows/build.yml)

### Benchmark (on Core i7-6700K)
```
$ roscd ndt_omp/data
$ rosrun ndt_omp align 251370668.pcd 251371071.pcd
--- pcl::NDT ---
single : 282.222[msec]
10times: 2921.92[msec]
fitness: 0.213937

--- pclomp::NDT (KDTREE, 1 threads) ---
single : 207.697[msec]
10times: 2059.19[msec]
fitness: 0.213937

--- pclomp::NDT (DIRECT7, 1 threads) ---
single : 139.433[msec]
10times: 1356.79[msec]
fitness: 0.214205

--- pclomp::NDT (DIRECT1, 1 threads) ---
single : 34.6418[msec]
10times: 317.03[msec]
fitness: 0.208511

--- pclomp::NDT (KDTREE, 8 threads) ---
single : 54.9903[msec]
10times: 500.51[msec]
fitness: 0.213937

--- pclomp::NDT (DIRECT7, 8 threads) ---
single : 63.1442[msec]
10times: 343.336[msec]
fitness: 0.214205

--- pclomp::NDT (DIRECT1, 8 threads) ---
single : 17.2353[msec]
10times: 100.025[msec]
fitness: 0.208511
```

Several methods for neighbor voxel search are implemented. If you select pclomp::KDTREE, results will be completely same as the original pcl::NDT. We recommend to use pclomp::DIRECT7 which is faster and stable. If you need extremely fast registration, choose pclomp::DIRECT1, but it might be a bit unstable.

<img src="data/screenshot.png" height="400pix" /><br>
Red: target, Green: source, Blue: aligned

## Related packages
- [ndt_omp](https://github.com/koide3/ndt_omp)
- [fast_gicp](https://github.com/SMRT-AIST/fast_gicp)

## Regression Test

### Preparation

You can use `script/convert_rosbag_to_test_data.py` to convert a rosbag to regression test data.

The regression test data should be placed in `./regression_test_data/input` directory.

```bash
./regression_test_data/
└── input
    ├── kinematic_state.csv
    ├── pointcloud_map.pcd  # means map.pcd
    └── sensor_pcd
        ├── pointcloud_00000000.pcd
        ├── pointcloud_00000001.pcd
        ├── pointcloud_00000002.pcd
        ├── ...
        ├── pointcloud_00001297.pcd
        ├── pointcloud_00001298.pcd
        └── pointcloud_00001299.pcd

2 directories, 1302 files
```

### build

```bash
mkdir build
cd build
cmake ..
make -j
```

### Run

```bash
./regression_test ../regression_test_data/input ../regression_test_data/output
```

### Check

```bash
python3 script/compare_regression_test_result.py ../regression_test_data/output ../regression_test_data/reference_output
```
