#!/bin/bash

gst-launch-1.0 souphttpsrc is-live=true location=$(pbpaste) ! hlsdemux ! tsdemux ! queue ! h264parse config-interval=-1 ! splitmuxsink max-size-time=10000000000 reset-muxer=true async-finalize=true location=video%02d.mp4
