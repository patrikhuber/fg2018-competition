import pymesh
import compute_vertices_to_mesh_distances as fg

# Load a ground truth scan and its landmark annotations:
groundtruth_scan = pymesh.load_mesh('F1008_A.obj') # from the challenge dataset
grundtruth_landmark_points = fg.read_groundtruth('F1008_A_landmarks.txt')

# Load your predicted mesh - the script needs a list of vertices and faces (triangles):
predicted_mesh = pymesh.load_mesh('your_mesh.obj')
# Load your annotations: The 3D coordinates of the 7 defined landmarks (see the README.md)
predicted_mesh_landmark_points = [ # These are example values!
    [-46.166801, 34.721901, -35.938000], # right eye outer corner
    [-18.926001, 31.543200, -29.964100], # right eye inner corner
    [19.257401, 31.576700, -30.229000], # left eye inner corner
    [46.191399, 34.452000, -36.131699], # left eye outer corner
    [-0.134549, -12.905100, -10.940000], # bottom of the nose
    [-22.698500, -35.266102, -27.407700], # right mouth corner
    [22.639299, -35.315800, -27.732201] # left mouth corner
]

# Compute the errors and save to a file:
out_file = "F1008_A_distances.txt"
fg.compute_vertices_to_mesh_distances(groundtruth_scan.vertices, grundtruth_landmark_points, predicted_mesh.vertices,
                                      predicted_mesh.faces, predicted_mesh_landmark_points, out_file)
print("Computed and saved distances.")
