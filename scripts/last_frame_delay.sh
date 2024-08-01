#!/bin/bash

# Input arguments
input_gif=$1
delay=$2  # delay in hundredths of a second
output_gif=$3

# Create a temporary directory to store frames
temp_dir=$(mktemp -d)

# Extract frames from the input GIF
convert +adjoin $input_gif $temp_dir/frame_%04d.gif

# Get the total number of frames
total_frames=$(ls $temp_dir | wc -l)
last_frame_index=$(($total_frames - 1))
last_frame=$(printf "$temp_dir/frame_%04d.gif" $last_frame_index)

# Modify the delay of the last frame
convert $last_frame -delay $delay $last_frame

# Reassemble the frames into the output GIF
convert -delay 10 $temp_dir/frame_*.gif -loop 0 $output_gif

# Clean up temporary directory
rm -rf $temp_dir

echo "Added a delay of $delay to the last frame of the GIF."
