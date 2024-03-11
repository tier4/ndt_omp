#!/bin/bash
set -eux

cd $(dirname $0)/../build

make -j8
./regression_test ../regression_test_data/input/ ../regression_test_data/output
python3 ../script/compare_regression_test_result.py ../regression_test_data/output/ ../regression_test_data/reference/
