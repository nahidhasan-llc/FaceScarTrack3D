import os
import numpy as np
import open3d as o3d

def main():
    print("--- Project 1: 3D Data Exploration with Open3D ---")
    
    # 1. Create artificial 3D Point Cloud data (A Sphere)
    # This simulates receiving raw X, Y, Z coordinate locations from Jen's scanner
    print("Generating a 3D sphere geometry sample...")
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    
    # Extract the raw spatial point array from the surface vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    
    print(f"Total points generated in space: {len(pcd.points)}")
    
    # 2. Programmatically manipulate 3D Properties
    # Let's paint the point cloud a bright Na'vi Avatar Cyan/Blue color using RGB values (scaled 0.0 to 1.0)
    print("Painting the 3D data points...")
    pcd.paint_uniform_color([0.2, 0.6, 0.9]) 
    
    # 3. Simulate Data Cleaning (Voxel Downsampling)
    # Scanners often provide way too much data or noisy data. We thin it out smoothly.
    print("Performing voxel downsampling to clean up point density...")
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.1)
    print(f"Cleaned up point count: {len(downsampled_pcd.points)}")
    
    # 4. Launch Interactive Spatial Viewer Window
    print("\nLaunching 3D Viewer...")
    print("💡 CONTROLS IN WINDOW:")
    print("   - Left Click + Drag : Rotate the 3D object")
    print("   - Shift + Left Click + Drag : Pan/Translate the workspace environment")
    print("   - Scroll Wheel : Zoom in / Zoom out")
    print("   - Press 'Q' : Close viewer windows cleanly")
    print("----------------------------------------------------------------")
    
    o3d.visualization.draw_geometries(
        [downsampled_pcd], 
        window_name="Open3D - Spatial Point Cloud Manipulation",
        width=800,
        height=600
    )

if __name__ == "__main__":
    main()