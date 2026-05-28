import open3d as o3d
import numpy as np

def main():
    print("--- Project 1.2: Upgrading to 3D Surface Meshes ---")
    
    # 1. Generate a solid 3D cylinder mesh
    # This creates actual faces (triangles) connecting the points, mimicking a face scan
    print("Generating a solid 3D cylinder mesh...")
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=2.0)
    
    # Compute normals (essential for the computer to calculate realistic shadows and lighting)
    mesh.compute_vertex_normals()
    
    # 2. Paint the mesh a solid clinical color (Light Grey/Teal)
    mesh.paint_uniform_color([1, 0, 1])
    
    print(f"Mesh Structure:")
    print(f"  - Total Vertices (Points): {len(mesh.vertices)}")
    print(f"  - Total Triangles (Faces): {len(mesh.triangles)}")
    
    # 3. Launch the 3D Visualizer
    print("\nLaunching solid mesh viewer...")
    print("💡 TIP: Notice how it looks solid and responds to lighting as you rotate it!")
    print("Press 'Q' inside the window to close it cleanly.")
    print("----------------------------------------------------------------")
    
    o3d.visualization.draw_geometries(
        [mesh], 
        window_name="Open3D - Solid Surface Mesh",
        width=800,
        height=600,
        mesh_show_wireframe=False  # Change to True if you want to see the triangle grid lines!
    )

if __name__ == "__main__":
    main()