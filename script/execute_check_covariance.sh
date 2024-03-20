#!/bin/bash
set -eux

# define function
function execute() {
    local INPUT_DIR=$(readlink -f $1)
    local OUTPUT_DIR=$(readlink -f $2)
    rm -rf ${OUTPUT_DIR}
    ./check_covariance ${INPUT_DIR} ${OUTPUT_DIR}
    python3 ../script/plot_covariance.py ${OUTPUT_DIR}/result.csv
    local TARGET_DIR=${OUTPUT_DIR}/covariance_each_frame
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
execute ../check_covariance_data/input_awsim_nishishinjuku_flat/ ../check_covariance_data/output_awsim_nishishinjuku_flat
execute ../check_covariance_data/input_awsim_nishishinjuku/ ../check_covariance_data/output_awsim_nishishinjuku
execute ../check_covariance_data/input_tunnel/ ../check_covariance_data/output_tunnel
