#!/bin/bash

# Check if URL is provided as parameter
if [ $# -eq 0 ]; then
    echo "Error: YouTube URL is required"
    echo "Usage: $0 <youtube_url>"
    exit 1
fi

# Get the URL from yt-dlp and pass it directly to gst-launch
VIDEO_URL=$(yt-dlp -f '[height<=720]' --get-url "$1")
echo $VIDEO_URL
gst-launch-1.0 souphttpsrc is-live=true location="$VIDEO_URL" ! hlsdemux ! tsdemux ! queue ! h264parse config-interval=-1 ! splitmuxsink max-size-time=10000000000 reset-muxer=true async-finalize=true location=video%02d.mp4
