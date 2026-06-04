import cv2
import numpy as np
import yaml


class Camera:
    def __init__(self, camera_param_file) -> None:
        with open(camera_param_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
            self.cx = data["cx"]
            self.cy = data["cy"]
            self.fx = data["fx"]
            self.fy = data["fy"]
            self.distortion_model = data.get("distortion_model", "standard")
            raw_dist = np.array(data["distCoeffs"]).flatten()

            self.cameraMatrix = np.array(
                [self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1]).reshape(3, 3)

            if self.distortion_model == "kb4":
                # Keep raw coefficients for fisheye undistortion only.
                # Expose zero distortion to solvePnP / GTSAM — corners are
                # pre-undistorted to the pinhole frame before use.
                self._fisheye_dist = raw_dist.reshape(4, 1)
                self.distCoeffs = np.zeros(4)
                self.k1 = self.k2 = self.p1 = self.p2 = 0.0
            else:
                self.distCoeffs = raw_dist
                if raw_dist.shape[0] >= 4:
                    self.k1 = raw_dist[0]
                    self.k2 = raw_dist[1]
                    self.p1 = raw_dist[2]
                    self.p2 = raw_dist[3]

        print(f"Camera parameters loaded from {camera_param_file}"
              f"\nfx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}"
              f", distortion_model={self.distortion_model}"
              f", distCoeffs={self.distCoeffs}")

    def undistort_points(self, pts):
        """Return pts undistorted into the pinhole pixel frame.

        For kb4 (Kannala-Brandt fisheye), maps raw detected pixel coordinates
        to the equivalent pinhole pixel coordinates so that standard PnP and
        GTSAM reprojection (zero distortion) can be used directly.
        For standard models, returns pts unchanged (distortion handled later).
        """
        if self.distortion_model != "kb4":
            return pts
        arr = np.array(pts, dtype=np.float64).reshape(-1, 1, 2)
        undist = cv2.fisheye.undistortPoints(
            arr, self.cameraMatrix, self._fisheye_dist, P=self.cameraMatrix)
        return undist.reshape(-1, 2)
