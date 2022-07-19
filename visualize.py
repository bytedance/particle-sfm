"""
Open3D visualizer for COLMAP format
Borrowed and Modified from the script written by Paul-Edouard Sarlin (paul.edouard.sarlin@gmail.com)
"""
import os
from copy import deepcopy
import numpy as np
import open3d as o3d
import open3d.visualization as vis
import pycolmap

print(o3d.__version__)
os.environ['WEBRTC_PORT'] = '8890'

def pcd_from_colmap(rec, min_track_length=4, max_reprojection_error=8):
    points = []
    colors = []
    for p3D in rec.points3D.values():
        if p3D.track.length() < min_track_length:
            continue
        if p3D.error > max_reprojection_error:
            continue
        points.append(p3D.xyz)
        colors.append(p3D.color/255.)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(points))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(colors))
    return pcd

def main(path, visualize=False):
    rec = pycolmap.Reconstruction(path)
    print(rec.summary())
    if not visualize:
        return

    app = vis.gui.Application.instance
    app.initialize()
    w = vis.O3DVisualizer(width=2048, height=1024)
    w.show_ground = False
    w.show_axes = False
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    # Add sparse point cloud
    pcd = pcd_from_colmap(rec)
    w.add_geometry('pcd', pcd, mat)

    # Define the camera frustums as lines
    camera_lines = {}
    for camera in rec.cameras.values():
        camera_lines[camera.camera_id] = o3d.geometry.LineSet.create_camera_visualization(
            camera.width, camera.height, camera.calibration_matrix(), np.eye(4), scale=0.3)

    # Draw the frustum for each image
    for image in rec.images.values():
        T = np.eye(4)
        T[:3, :4] = image.inverse_projection_matrix()
        cam = deepcopy(camera_lines[image.camera_id]).transform(T)
        cam.paint_uniform_color([1.0, 0.0, 0.0])  # red
        w.add_geometry(image.name, cam)

    w.reset_camera_to_default()
    w.scene_shader = w.UNLIT
    w.point_size = 1
    w.enable_raw_mode(True)
    app.add_window(w)
    app.run()

if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser(description="visualization of colmap reconstruction")
    arg_parser.add_argument("-i", "--input_dir", type=str, required=True, help="input path")
    arg_parser.add_argument("--visualize", action="store_true", help="whether to visualize")
    args = arg_parser.parse_args()
    main(args.input_dir, visualize=args.visualize)

