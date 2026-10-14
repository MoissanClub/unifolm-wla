"""Read RGB from the existing video publisher without opening its camera.

One worker continuously decodes the stream while inference runs. Timestamps
mark local decode completion, not camera exposure or sensor synchronization.
OpenCV/FFmpeg retain their native connection and read behavior: this module
adds no capture deadline. Closing waits for a pending backend operation to
return before that worker releases the capture.
"""
from __future__ import annotations

import os
import threading
import time


class HeadStream:
    def __init__(self, url):
        import cv2

        self.url = str(url)
        # FFmpeg capture options are process-wide. Preserve an explicit user
        # configuration; default this deployment's RTSP reader to TCP.
        if self.url.lower().startswith(("rtsp://", "rtsps://")):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self._cv2 = cv2
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._frame = None
        self._decode_monotonic = None
        self._sequence = self._returned_sequence = 0
        self._error = None
        self._thread = threading.Thread(target=self._run, name="wla-head-stream")
        self._thread.start()

    def _run(self):
        capture = None
        try:
            if self._stop.is_set():
                return
            capture = self._cv2.VideoCapture(self.url, self._cv2.CAP_FFMPEG)
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open head stream: {self.url}")
            while not self._stop.is_set():
                ok, bgr = capture.read()
                decoded = time.monotonic()
                if self._stop.is_set():
                    break
                if not ok or bgr is None:
                    raise RuntimeError(f"Head stream decode failed: {self.url}")
                rgb = bgr[..., ::-1].copy()
                with self._condition:
                    self._frame = rgb
                    self._decode_monotonic = decoded
                    self._sequence += 1
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:
                    with self._condition:
                        if self._error is None:
                            self._error = exc
                        self._condition.notify_all()

    def read(self):
        """Return the latest RGB frame newer than this consumer's previous read.

        Intermediate decoded frames are replaced while inference is busy; they
        are not queued for later playback. A sequence number is local to this
        decoder and does not imply a camera frame ID or an exposure timestamp.
        """
        with self._condition:
            while True:
                if self._stop.is_set():
                    raise RuntimeError("Head stream is closed")
                if self._error is not None:
                    raise RuntimeError(f"Head stream failed: {self._error}") from self._error
                if self._sequence > self._returned_sequence:
                    self._returned_sequence = self._sequence
                    return self._frame, {
                        "source": self.url,
                        "decode_monotonic": self._decode_monotonic,
                        "sequence": self._sequence,
                        "timestamp_kind": "local_decode_completion_not_exposure",
                    }
                self._condition.wait()

    def close(self):
        """Stop and join; only the worker releases the OpenCV capture.

        If open/read is blocked, shutdown waits for FFmpeg's existing transport
        behavior. Releasing from this caller concurrently with read is avoided.
        """
        with self._condition:
            self._stop.set()
            self._condition.notify_all()
        self._thread.join()
