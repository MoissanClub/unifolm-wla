import os
import queue
import sys
import threading
import time
from concurrent.futures import Future, TimeoutError
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.g1.camera import HeadStream


def background(function):
    result = Future()

    def invoke():
        try:
            result.set_result(function())
        except BaseException as exc:
            result.set_exception(exc)

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    return result


@pytest.fixture
def stream_factory(monkeypatch):
    captures, streams = [], []

    class Capture:
        def __init__(self, url, backend):
            self.url, self.backend = url, backend
            self.frames = queue.Queue()
            self.reads = queue.Queue()
            self.released = threading.Event()
            self.owner = threading.get_ident()
            self.release_thread = None
            captures.append(self)
            created.set()

        def isOpened(self):
            return opened[0]

        def read(self):
            assert threading.get_ident() == self.owner
            self.reads.put(True)
            return self.frames.get()

        def release(self):
            self.release_thread = threading.get_ident()
            self.released.set()

    created = threading.Event()
    opened = [True]
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=Capture, CAP_FFMPEG=1900))

    def make(*, can_open=True, url="rtsp://127.0.0.1:8554/d435"):
        opened[0] = can_open
        created.clear()
        stream = HeadStream(url)
        streams.append(stream)
        assert created.wait(1)
        return stream, captures[-1]

    yield make
    for capture in captures:
        capture.frames.put((False, None))
    for stream in streams:
        stream.close()


def frame(value):
    return np.full((3, 4, 3), value, dtype=np.uint8)


def publish(capture, bgr):
    # Reaching the next read means the preceding frame has been published.
    capture.frames.put((True, bgr))
    capture.reads.get(timeout=1)


def test_rgb_order_and_decode_timestamp(stream_factory):
    stream, capture = stream_factory()
    capture.reads.get(timeout=1)
    before = time.monotonic()
    bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    publish(capture, bgr)
    rgb, metadata = stream.read()
    np.testing.assert_array_equal(rgb, [[[30, 20, 10], [60, 50, 40]]])
    assert not np.shares_memory(rgb, bgr)
    assert before <= metadata["decode_monotonic"] <= time.monotonic()
    assert metadata["sequence"] == 1
    assert metadata["source"] == capture.url
    assert metadata["timestamp_kind"] == "local_decode_completion_not_exposure"
    assert capture.backend == 1900


def test_reader_drains_stream_while_consumer_is_busy(stream_factory):
    stream, capture = stream_factory()
    capture.reads.get(timeout=1)
    publish(capture, frame(1))
    _, first = stream.read()
    publish(capture, frame(2))
    publish(capture, frame(3))
    rgb, latest = stream.read()
    assert first["sequence"] == 1
    assert latest["sequence"] == 3
    assert (rgb == 3).all()


def test_read_waits_for_a_new_sequence(stream_factory):
    stream, capture = stream_factory()
    capture.reads.get(timeout=1)
    publish(capture, frame(1))
    stream.read()
    waiting = background(stream.read)
    with pytest.raises(TimeoutError):
        waiting.result(timeout=0.05)
    publish(capture, frame(2))
    rgb, metadata = waiting.result(timeout=1)
    assert metadata["sequence"] == 2
    assert (rgb == 2).all()


def test_open_failure_releases_capture_and_reaches_caller(stream_factory):
    stream, capture = stream_factory(can_open=False)
    with pytest.raises(RuntimeError, match="Cannot open head stream"):
        stream.read()
    assert capture.released.wait(1)


def test_decode_failure_does_not_return_a_cached_frame(stream_factory):
    stream, capture = stream_factory()
    capture.reads.get(timeout=1)
    publish(capture, frame(1))
    capture.frames.put((False, None))
    assert capture.released.wait(1)
    with pytest.raises(RuntimeError, match="decode failed"):
        stream.read()


def test_close_waits_for_worker_and_never_releases_during_read(stream_factory):
    stream, capture = stream_factory()
    capture.reads.get(timeout=1)
    waiting_reader = background(stream.read)
    closing = background(stream.close)
    assert stream._stop.wait(1)
    with pytest.raises(RuntimeError, match="closed"):
        waiting_reader.result(timeout=1)
    assert not closing.done()
    assert not capture.released.is_set()
    capture.frames.put((False, None))
    closing.result(timeout=1)
    assert capture.released.is_set()
    assert capture.release_thread == capture.owner
    assert not stream._thread.is_alive()
    stream.close()  # Repeated shutdown is harmless.


def test_rtsp_defaults_to_tcp_without_overwriting_user_options(stream_factory, monkeypatch):
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    stream_factory()
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"


def test_explicit_ffmpeg_options_are_preserved(stream_factory, monkeypatch):
    options = "rtsp_transport;udp|fflags;nobuffer"
    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", options)
    stream_factory()
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == options
