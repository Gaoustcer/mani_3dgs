import open3d as o3d
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pcd-path",type = str)
args = parser.parse_args()
pcd_path = args.pcd_path
pcd = o3d.io.read_point_cloud(pcd_path)
o3d.draw_geometries([pcd])