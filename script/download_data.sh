#!/bin/bash
set -eux

cd $(dirname $0)/..

mkdir -p regression_test_data
cd regression_test_data

# Download the input data
wget 'https://drive.google.com/uc?export=download&id=1E-_zj2nchmntioSJJgyoDQEYHtrs3o-C' -O ndt_omp_regression_test_input.zip --quiet
unzip -q ndt_omp_regression_test_input.zip
rm ndt_omp_regression_test_input.zip

# Download the map data
wget https://github.com/tier4/AWSIM/releases/download/v1.1.0/nishishinjuku_autoware_map.zip --quiet
unzip -q nishishinjuku_autoware_map.zip
rm nishishinjuku_autoware_map.zip
mv nishishinjuku_autoware_map/pointcloud_map.pcd ./input/
rm -rf nishishinjuku_autoware_map
