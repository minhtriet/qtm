#!/bin/bash

# Input arguments
input_gif=$1
desired_duration=$2  # in seconds
output_gif=$3

# Get the total frame count of the input GIF
total_frames=$(ffmpeg -i $input_gif 2>&1 | grep -oP "(?<=, )[0-9]+ fps" | awk '{print $1}')

# Calculate the new frame rate
new_fps=$(echo "$total_frames / $desired_duration" | bc -l)

# Change the frame rate of the GIF
ffmpeg -i $input_gif -filter:v "setpts=N/(FPS*TB)" -r $new_fps -y $output_gif

echo "GIF successfully changed to end in $desired_duration seconds."
