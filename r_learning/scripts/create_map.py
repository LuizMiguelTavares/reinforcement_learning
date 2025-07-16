#!/usr/bin/env python3
"""
Interactive grid‑map creator for ROS 1 (Noetic).

 • Click on a cell to toggle an obstacle (red = occupied).
 • Close the window to finish. The node will:
      – publish a nav_msgs/OccupancyGrid (one‑shot)
      – write <file_name>.pgm and <file_name>.yaml
"""

import os
import yaml
import rospy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header


def draw_obs_grid(nx: int, ny: int) -> np.ndarray:
    """Returns a (ny × nx) numpy array with 1 = obstacle, 0 = free."""
    grid = np.zeros((ny, nx), dtype=int)
    fig, ax = plt.subplots(figsize=(8, 8))

    def redraw():
        ax.clear()
        ax.imshow(grid, cmap='Reds', vmin=0, vmax=1, origin='lower')
        ax.set_xticks(np.arange(-.5, nx, 1), minor=True)
        ax.set_yticks(np.arange(-.5, ny, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", size=0)
        ax.set_xticks(np.arange(0, nx, 1))
        ax.set_yticks(np.arange(0, ny, 1))
        ax.set_title("Left‑click to toggle obstacles → close window to save")
        fig.canvas.draw_idle()

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= ix < nx and 0 <= iy < ny:
            grid[iy, ix] = 1 - grid[iy, ix]
            redraw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    redraw()
    plt.show()
    return grid


def numpy_to_occ(grid: np.ndarray) -> np.ndarray:
    """Return flattened int8 array in ROS occupancy style (0/100)."""
    occ = np.where(grid == 1, 100, 0).astype(np.int8)
    return occ.flatten(order='C')  # row‑major, bottom row first (origin='lower')


def save_as_pgm_yaml(grid: np.ndarray, res: float, file_stem: str):
    """
    Write ${file_stem}.pgm and ${file_stem}.yaml compatible with map_server.
    Obstacles → black (0), free → white (254), unknown → grey (205)
    """
    pgm_path = f"{file_stem}.pgm"
    yaml_path = f"{file_stem}.yaml"

    # PGM expects rows from top to bottom. Flip vertically.
    img = np.where(grid == 1, 0, 254).astype(np.uint8)
    img = np.flipud(img)
    Image.fromarray(img, mode='L').save(pgm_path)
    rospy.loginfo(f"[obs_grid_creator] Wrote {pgm_path}")

    meta = {
        "image": os.path.basename(pgm_path),
        "resolution": res,
        "origin": [0.0, 0.0, 0.0],   # x, y, yaw (rad)
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False)
    rospy.loginfo(f"[obs_grid_creator] Wrote {yaml_path}")


def make_occupancy_grid_msg(grid: np.ndarray,
                            res: float,
                            frame: str = "map") -> OccupancyGrid:
    h, w = grid.shape
    og = OccupancyGrid()
    og.header = Header(frame_id=frame, stamp=rospy.Time.now())
    og.info = MapMetaData(
        resolution=res,
        width=w,
        height=h,
        origin=Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)),
        map_load_time=rospy.Time.now(),
    )
    og.data = list(numpy_to_occ(grid))
    return og


def main():
    rospy.init_node("obs_grid_creator")

    # ---- Parameters ----
    nx          = rospy.get_param("~grid_width", 9)
    ny          = rospy.get_param("~grid_height", 7)
    cell_size   = rospy.get_param("~cell_size_cm", 10.0) / 100.0  # → m
    out_dir     = rospy.get_param("~output_dir", "/tmp/obs_grids")
    base_name   = rospy.get_param("~file_name", "mapa_de_obstaculos")
    frame_id    = rospy.get_param("~frame_id", "map")

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    file_stem = os.path.join(out_dir, base_name)

    rospy.loginfo(f"[obs_grid_creator] Click‑to‑draw {nx}×{ny} grid")
    grid = draw_obs_grid(nx, ny)  # blocks until window is closed

    # ---- Publish once (handy for RViz preview) ----
    pub = rospy.Publisher("~occupancy_grid", OccupancyGrid, queue_size=1, latch=True)
    og_msg = make_occupancy_grid_msg(grid, cell_size, frame_id)
    pub.publish(og_msg)
    rospy.loginfo("[obs_grid_creator] Published nav_msgs/OccupancyGrid on "
                  f"{pub.resolved_name}")

    # ---- Save files ----
    save_as_pgm_yaml(grid, cell_size, file_stem)

    rospy.loginfo("[obs_grid_creator] All done – shut down node.")
    rospy.sleep(1.0)


if __name__ == "__main__":
    main()