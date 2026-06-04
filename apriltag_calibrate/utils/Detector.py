import sys
import ctypes
import numpy as np

try:
    ctypes.CDLL("/usr/local/lib/libapriltag.so.3")
    for _py_ver in ("3.14", "3.13", "3.12", "3.11", "3.10"):
        _path = f"/usr/local/lib/python{_py_ver}/site-packages"
        if _path not in sys.path:
            sys.path.insert(0, _path)
except OSError:
    pass

import apriltag


class Detection:
    def __init__(self, d):
        self.tag_id = d["id"]
        self.corners = np.array(d["lb-rb-rt-lt"], dtype=np.float64)
        self.center = np.array(d["center"], dtype=np.float64)


class Detector:
    def __init__(self, family: str):
        self._detector = apriltag.apriltag(family)

    def detect(self, gray) -> list:
        return [Detection(d) for d in self._detector.detect(gray)]
