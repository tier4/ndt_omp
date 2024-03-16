#!/bin/bash
set -eux

# define function
function make_movie() {
    local TARGET_DIR=$1
    ffmpeg -r 10 \
           -i ${TARGET_DIR}/%08d.png \
           -vcodec libx264 \
           -pix_fmt yuv420p \
           -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
           -r 10 \
           ${TARGET_DIR}/../movie.mp4
}

#
# execute
#
cd $(dirname $0)/../build

make -j8
./check_covariance ../check_covariance_data/input_awsim_nishishinjuku_flat/ ../check_covariance_data/output_awsim_nishishinjuku_flat
python3 ../script/plot_covariance.py ../check_covariance_data/output_awsim_nishishinjuku_flat/result.csv
make_movie ../check_covariance_data/output_awsim_nishishinjuku_flat/covariance_each_frame

./check_covariance ../check_covariance_data/input_awsim_nishishinjuku/ ../check_covariance_data/output_awsim_nishishinjuku
python3 ../script/plot_covariance.py ../check_covariance_data/output_awsim_nishishinjuku/result.csv
make_movie ../check_covariance_data/output_awsim_nishishinjuku/covariance_each_frame

./check_covariance ../check_covariance_data/input_tunnel ../check_covariance_data/output_tunnel
python3 ../script/plot_covariance.py ../check_covariance_data/output_tunnel/result.csv
make_movie ../check_covariance_data/output_tunnel/covariance_each_frame
