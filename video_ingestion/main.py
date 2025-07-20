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
OUT_DIR = pathlib.Path("clips")
OUT_DIR.mkdir(exist_ok=True)

FPS           = 30          # used by camera → adjust caps as needed
MIN_MOTION_FR = 2           # >1 helps avoid spurious triggers
GAP_SEC       = 4           # how long w/o motion before we stop saving
THRESHOLD     = 0.03        # fraction of cells that must change
LOCATION_TPL  = str(OUT_DIR / "motion_%05d.mp4")

# ───── GStreamer init ───────────────────────────────────────────────────────
Gst.init(None)

pipeline_descr = f"""
    souphttpsrc is-live=true location={sys.argv[1]} ! hlsdemux ! tsdemux ! tee name=t

    t. ! h264parse ! decodebin ! videoconvert ! motioncells ! fakesink

    t. ! queue name=q
"""

pipeline = Gst.parse_launch(pipeline_descr)
tee        = pipeline.get_by_name('t')        # your tee
q          = pipeline.get_by_name('q')
rec_branch = None

def build_rec_branch():
    global rec_branch
    global q
    rec_branch = Gst.Bin.new(None)

    parse = Gst.ElementFactory.make('h264parse', 'parse')
    mux   = Gst.ElementFactory.make('splitmuxsink', 'sink')
    mux.set_property('location', LOCATION_TPL)
    mux.set_property('max-size-time', 10 * Gst.SECOND)

    for e in (parse, mux):
        rec_branch.add(e)
    parse.link(mux)

    # Ghost sink pad so we can link the bin to the tee
    # ghost = Gst.GhostPad.new('sink', q.get_static_pad('sink'))
    # rec_branch.add_pad(ghost)

    pipeline.add(rec_branch)
    rec_branch.sync_state_with_parent()

    queue_sink_pad = q.get_static_pad("sink")

    # request a new pad from tee and link
    tee_pad = tee.request_pad_simple('src_%u')
    queue_sink_pad.link(tee_pad)

def destroy_rec_branch():
    global rec_branch
    if not rec_branch:
        return
    sink = rec_branch.get_by_name('sink')

    def _finish(_sink, _pad):
        # unlink and free once muxer is done
        for ghost in rec_branch.iterate_pads():
            peer = ghost.get_peer()
            ghost.unlink(peer)
            tee.release_request_pad(peer)
        rec_branch.set_state(Gst.State.NULL)
        pipeline.remove(rec_branch)
        rec_branch = None

    # route EOS only to the recorder branch
    sink.get_static_pad('sink').send_event(Gst.Event.new_eos())
    sink.connect('eos', _finish)

# ───── bus handler ──────────────────────────────────────────────────────────
def on_bus_message(bus, msg, loop):
    if msg.type != Gst.MessageType.ELEMENT:
        return

    st = msg.get_structure()
    print(st.get_name())
    if not st or st.get_name() != "motion":
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
