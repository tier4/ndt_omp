#!/bin/bash
set -eux

INPUT_DIR=$(readlink -f $1)
OUTPUT_DIR=$(readlink -f $2)

cd $(dirname $0)/../build

make -j8

rm -rf ${OUTPUT_DIR}
./check_covariance ${INPUT_DIR} ${OUTPUT_DIR}
python3 ../script/plot_covariance.py ${OUTPUT_DIR}/result.csv
