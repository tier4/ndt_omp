#!/bin/bash
set -eux

cd $(dirname $0)/../build

make -j8
./check_covariance ../check_covariance_data/input_awsim_nishishinjuku/ ../check_covariance_data/output_awsim_nishishinjuku
python3 ../script/plot_covariance.py ../check_covariance_data/output_awsim_nishishinjuku/result.csv

./check_covariance ../check_covariance_data/input_tunnel ../check_covariance_data/output_tunnel
python3 ../script/plot_covariance.py ../check_covariance_data/output_tunnel/result.csv
