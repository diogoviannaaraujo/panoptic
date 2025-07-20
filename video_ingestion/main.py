#!/usr/bin/env python3
"""
Record MP4 clips *only* while motion is detected by the `motioncells` plugin.

$ pip install PyGObject
$ sudo apt install gstreamer1.0-libav   # for x264enc / mp4mux if not present
$ ./record_on_motion.py                 # ^C to quit
"""
import sys
import time
import gi, pathlib, datetime, os, sys, signal
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

# ───── configuration ────────────────────────────────────────────────────────

FPS           = 30          # used by camera → adjust caps as needed
MIN_MOTION_FR = 2           # >1 helps avoid spurious triggers
GAP_SEC       = 4           # how long w/o motion before we stop saving
THRESHOLD     = 0.03        # fraction of cells that must change

# ───── GStreamer init ───────────────────────────────────────────────────────
Gst.init(None)

#    souphttpsrc is-live=true location={sys.argv[1]} ! hlsdemux ! tsdemux ! tee name=t

pipeline_descr = f"""
    rtspsrc location=rtsp://100.96.79.118:8554/webCamStream protocols=tcp ! rtph264depay ! tee name=t

    t. ! h264parse ! decodebin !  videorate ! videoscale ! video/x-raw,width=320,height=240,framerate=5/1 ! videoconvert ! motioncells display=true ! fakesink
    t. ! queue name=q leaky=1 max-size-buffers=0 max-size-bytes=0 max-size-time={10 * Gst.SECOND}
"""

pipeline = Gst.parse_launch(pipeline_descr)
q          = pipeline.get_by_name('q')
rec_branch = None
old_branch = None
rec_stop = False

def build_rec_branch():
    global rec_branch
    global q
    rec_branch = Gst.Bin.new(None)

    dir = pathlib.Path("clips/" + str(int(time.time())))
    dir.mkdir(exist_ok=True)
    locationtpl  = str(dir / "%05d.mp4")

    rec_q = Gst.ElementFactory.make('queue', 'rec_q')
    rec_q.set_property('max-size-buffers', 0)
    rec_q.set_property('max-size-bytes', 0)
    rec_q.set_property('max-size-time', 10 * Gst.SECOND)
    rec_q.set_property('min-threshold-time', 4 * Gst.SECOND)
    parse = Gst.ElementFactory.make('h264parse', 'parse')
    mux   = Gst.ElementFactory.make('splitmuxsink', 'sink')
    mux.set_property('location', locationtpl)
    mux.set_property('max-size-time', 10 * Gst.SECOND)

    for e in (rec_q, parse, mux):
        rec_branch.add(e)
    rec_q.link(parse)
    parse.link(mux)

    # Ghost sink pad so we can link the bin to the tee
    ghost = Gst.GhostPad.new('sink', rec_q.get_static_pad('sink'))
    rec_branch.add_pad(ghost)

    pipeline.add(rec_branch)
    rec_branch.sync_state_with_parent()

    q.link(rec_branch)

def destroy_rec_branch():
    global rec_branch
    global q
    global rec_stop
    if not rec_branch:
        return
    rec_q = rec_branch.get_by_name('rec_q')
    sink = rec_branch.get_by_name('sink')
    sink.emit("split-after")

    # route EOS only to the recorder branch
    rec_q.get_static_pad('sink').send_event(Gst.Event.new_eos())
    print(f"[{datetime.datetime.now():%T}]  📧  Eos Sent")
    rec_stop = True

# ───── bus handler ──────────────────────────────────────────────────────────
def on_bus_message(bus, msg, loop):
    global rec_stop
    global rec_branch
    global old_branch
    global pipeline
    global q
    if msg.type != Gst.MessageType.ELEMENT:
        return

    st = msg.get_structure()
    name = st.get_name()
    print(name)

    if name == "splitmuxsink-fragment-closed" and rec_stop == True:
        print(f"[{datetime.datetime.now():%T}]  ⭐️  Finishing")
        q.unlink(rec_branch)
        rec_branch.set_state(Gst.State.NULL)
        pipeline.remove(rec_branch)
        old_branch = rec_branch
        rec_branch = None


    if not st:
        return

    if st.has_field("motion_begin"):
        print(f"[{datetime.datetime.now():%T}]  🔴  Motion START")
        if rec_branch is None:
            build_rec_branch()
    elif st.has_field("motion_finished"):
        print(f"[{datetime.datetime.now():%T}]  ⚪️  Motion END")
        if rec_branch:
            destroy_rec_branch()


bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_bus_message, None)

# ───── run ──────────────────────────────────────────────────────────────────
loop = GLib.MainLoop()

def shutdown(*_):
    pipeline.set_state(Gst.State.NULL)
    loop.quit()

signal.signal(signal.SIGINT, shutdown)
pipeline.set_state(Gst.State.PLAYING)
print("Recording only while motion is detected …  Ctrl-C to quit.")
loop.run()
