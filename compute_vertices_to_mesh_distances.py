import numpy as np
import pymesh
from math import sqrt
import csv


def compute_vertices_to_mesh_distances(groundtruth_vertices, grundtruth_landmark_points, predicted_mesh_vertices,
                                       predicted_mesh_faces,
                                       predicted_mesh_landmark_points, out_filename):
    """
    This script computes the reconstruction error between an input mesh and a ground truth mesh.
    :param groundtruth_vertices: An n x 3 numpy array of vertices from a ground truth scan.
    :param grundtruth_landmark_points: A 7 x 3 list with annotations of the ground truth scan.
    :param predicted_mesh_vertices: An m x 3 numpy array of vertices from a predicted mesh.
    :param predicted_mesh_faces: A k x 3 numpy array of vertex indices composing the predicted mesh.
    :param predicted_mesh_landmark_points: A 7 x 3 list containing the annotated 3D point locations in the predicted mesh.
    :param out_filename: Filename to write the resulting distances to (e.g. F1008_A_distances.txt).
    :return: A list of distances (errors), one for each vertex in the groundtruth mesh, and the associated vertex index in the ground truth scan.

    The grundtruth_landmark_points and predicted_mesh_landmark_points have to contain points in the following order:
    (1) right eye outer corner, (2) right eye inner corner, (3) left eye inner corner, (4) left eye outer corner,
    (5) nose bottom, (6) right mouth corner, (7) left mouth corner.
    """

    # Do procrustes based on the 7 points:
    # The ground truth scan is in mm, so by aligning the prediction to the ground truth, we get meaningful units.
    d, Z, tform = procrustes(np.array(grundtruth_landmark_points), np.array(predicted_mesh_landmark_points),
                             scaling=True, reflection='best')
    # Use tform to transform all vertices in predicted_mesh_vertices to the ground truth reference space:
    predicted_mesh_vertices_aligned = []
    for v in predicted_mesh_vertices:
        s = tform['scale']
        R = tform['rotation']
        t = tform['translation']
        transformed_vertex = s * np.dot(v, R) + t
        predicted_mesh_vertices_aligned.append(transformed_vertex)

    # Compute the mask: A circular area around the center of the face. Take the nose-bottom and go upwards a bit:
    nose_bottom = np.array(grundtruth_landmark_points[4])
    nose_bridge = (np.array(grundtruth_landmark_points[1]) + np.array(
        grundtruth_landmark_points[2])) / 2  # between the inner eye corners
    face_centre = nose_bottom + 0.3 * (nose_bridge - nose_bottom)
    # Compute the radius for the face mask:
    outer_eye_dist = np.linalg.norm(np.array(grundtruth_landmark_points[0]) + np.array(grundtruth_landmark_points[3]))
    nose_dist = np.linalg.norm(nose_bridge - nose_bottom)
    mask_radius = 1.5 * (outer_eye_dist + nose_dist) / 2

    # Find all the vertex indices in the ground truth scan that lie within the mask area:
    vertex_indices_mask = []  # vertex indices in the source mesh (the ground truth scan)
    points_on_groundtruth_scan_to_measure_from = []
    for vertex_idx, vertex in enumerate(groundtruth_vertices):
        dist = np.linalg.norm(vertex - face_centre) # We use Euclidean distance for the mask area for now.
        if dist <= mask_radius:
            vertex_indices_mask.append(vertex_idx)
            points_on_groundtruth_scan_to_measure_from.append(vertex)
    assert len(vertex_indices_mask) == len(points_on_groundtruth_scan_to_measure_from)

    # For each vertex on the ground truth mesh, find the closest point on the surface of the predicted mesh:
    predicted_mesh_pymesh = pymesh.meshio.form_mesh(np.array(predicted_mesh_vertices_aligned), predicted_mesh_faces)
    squared_distances, face_indices, closest_points = pymesh.distance_to_mesh(predicted_mesh_pymesh,
                                                                              points_on_groundtruth_scan_to_measure_from)
    distances = [sqrt(d2) for d2 in squared_distances]

    # Save the distances to a file, alongside with each vertex id of the ground truth scan that the distance has been computed for:
    with open(out_filename, 'w') as csv_file:
        wr = csv.writer(csv_file, delimiter=' ')
        vertex_indices_with_distances = [[v_idx, v] for v_idx, v in zip(vertex_indices_mask, distances)]
        wr.writerows(vertex_indices_with_distances)


def read_groundtruth(filename):
    """
    Reads a file with 7 annotations for a ground truth scan.
    :param filename: Filename of the text file to load.
    :return: A list of 7 annotations in the order (1) right eye outer corner, (2) right eye inner corner, (3) left eye inner corner, (4) left eye outer corner,
             (5) nose bottom, (6) right mouth corner, (7) left mouth corner.
    """
    groundtruth_points = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if row:  # there might be an empty line at the end of the file
                groundtruth_points.append([float(row[1]), float(row[2]), float(row[3])])
    if len(groundtruth_points) != 7:
        raise Exception("Error: Expected 7 landmarks in the given ground truth file " + filename + ".")
    return groundtruth_points


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Code from: https://stackoverflow.com/a/18927641.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform
