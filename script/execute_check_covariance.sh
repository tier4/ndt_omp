#!/bin/bash
set -eux

cd $(dirname $0)/../build

make -j8
./check_covariance ../check_covariance_data/input/ ../check_covariance_data/output
python3 ../script/plot_covariance.py ../check_covariance_data/output/result.csv
