import numpy as np
from scipy.sparse import lil_matrix
from mesh_ops import MeshOps
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import scipy as sp


def assemble_poisson(mesh: MeshOps, param):

    num_elements = mesh.getNumberOfTriangles()
    # Shape functions on P1 reference element, assuming counter clockwise from (0,0)
    p1_points = [[0, 0], [1, 0], [0, 1]]
    shape_1 = [lambda x, y: 1 - x - y, lambda x, y: x, lambda x, y: y]
    dshape_1 = [lambda x, y: [-1, -1], lambda x, y: [1, 0], lambda x, y: [0, 1]]
    # Shape functions on P2 reference element, assuming counter clockwise from (0,0)
    p2_points = [[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]]
    shape_2 = [
        lambda x, y: 1 - 3 * x - 3 * y + 4 * x * y + 2 * x**2 + 2 * y**2,
        lambda x, y: -x + 2 * x**2,
        lambda x, y: -y + 2 * y**2,
        lambda x, y: 4 * x - 4 * x * y - 4 * x**2,
        lambda x, y: 4 * x * y,
        lambda x, y: 4 * y - 4 * x * y - 4 * y**2,
    ]
    dshape_2 = [
        lambda x, y: [4 * x + 4 * y - 3, 4 * y + 4 * x - 3],
        lambda x, y: [4 * x - 1, 0],
        lambda x, y: [0, 4 * y - 1],
        lambda x, y: [4 - 4 * y - 8 * x, -4 * x],
        lambda x, y: [4 * y, 4 * x],
        lambda x, y: [-4 * y, 4 - 4 * x - 8 * y],
    ]

    # points for quadrature on reference element from hw5
    source = param["source"]
    order = param_poisson["order"]
    if order == 2:
        phi = shape_2
        dphi = dshape_2
        local_n = 6
        phi_points = p2_points
        N = len(mesh.points)
        wts, pts, N_quadr = mesh.IntegrationRuleOfTriangle()

    else:
        phi = shape_1
        dphi = dshape_1
        local_n = 3
        phi_points = p1_points
        N = mesh.getNumberNodes()  # only works when not not updating mesh.nbNod
        pts = [[1 / 6, 1 / 6], [4 / 6, 1 / 6], [1 / 6, 4 / 6]]
        wts = [1 / 6] * 6
        N_quadr = 3

    A = lil_matrix((N, N))
    f = np.zeros(N)

    dphi_ref = np.array([[dphi[i](x, y) for x, y in pts] for i in range(local_n)])
    # Iterate over all elements
    for e in range(num_elements):
        ########################
        # calculate contributions of every element to each integral for the
        # shape functions on the vertecies of the element  (for bilinear form / matrix)
        #########################3
        elemA = np.zeros((local_n, local_n))

        invJ = mesh.calcInverseJacobianOfTriangle(e)
        J = mesh.calcJacobianOfTriangle(e)
        # calculates (B_K ^-T grad phi_n)^T = grad phi_n^T B_K^-1
        # this is done all at once by stacking the transposed gradients on top of each other
        # to a 3x2 matrix resulting in stacked result row vectors in a 3x2 matrix
        for i in range(local_n):
            pass

        dphi = dphi_ref @ invJ
        detJ = mesh.calcJacobianDeterminantOfTriangle(e)
        # (dphi @ dphi.T) will calculate a matrix with a_ij = phi_i * phi_j <- dot product
        # 1/2 because the integral is constant, pulling everything out leaves the area as a factor
        # Area of a triangle with sidelenghts 1 is 1/2
        elemA = (dphi @ dphi.T) * detJ * 1 / 2
        con = mesh.getNodeNumbersOfTriangle(e, order=order)
        # np.ix_(a,b) will create indexes p,q which, when indexing a matrix with it will select
        # every element in rows a that are at columns b
        # eg A([1,3,5],[6,7,8]) will select (1,6) (1,7) (3,8) (3,6) (3,7) (5,8) (5,6) (5,7) (1,8)
        # here we acess the elements corresponding to the node numbers
        A[np.ix_(con, con)] += elemA

        ###########################
        # calculate contributions of each element to the integrals of the linear functional
        # for each shape function on the vertices
        ###########################
        n1 = con[0]
        p1 = mesh.points[n1]

        for i, node in enumerate(con):
            temp = 0
            for j in range(N_quadr):
                pt = pts[j]
                x = p1 + J @ pt
                temp += wts[j] * phi[i](pt[0], pt[1]) * source(x[0], x[1])
            f[node] += temp * detJ

    return A, f


def apply_bc_poisson(mesh: MeshOps, A, f, param):

    def on_neuman(x, y):
        return x == 1

    def on_dirichlet(x, y):
        return x == 0 or y == 0 or y == 1

    dirichlet = param["dirichlet"]
    neumann = param["neumann"]
    for i in range(len(f)):
        node = mesh.points[i]
        x, y = node
        # Apply Dirichlet boundary conditions
        if on_dirichlet(x, y):
            f[i] = dirichlet
            A[i, :] = 0
            # A[:, i] = 0
            A[i, i] = 1

    return A, f


def add_triangle6from3(mesh: MeshOps):

    extra_points = []
    points_idx = mesh.getNumberNodes() - 1
    triangles = []
    # map that maps edge to new edge midpoint in point index
    vert_p_map = {}
    points = mesh.getNodeList()
    # get all triangles
    for e in range(mesh.getNumberOfTriangles()):
        # extract all edges
        con = mesh.getNodeNumbersOfTriangle(e)
        edge1 = (con[0], con[1])
        edge2 = (con[1], con[2])
        edge3 = (con[2], con[0])

        midpoints = []
        # assemble new connectivity list for triangle6
        for edge in [edge1, edge2, edge3]:
            ep1_ix, ep2_ix = edge
            sorted_edge = tuple(sorted(edge))

            if sorted_edge in vert_p_map.keys():
                emp = vert_p_map[sorted_edge]
            else:
                # add new point to point list
                points_idx += 1
                vert_p_map[sorted_edge] = points_idx
                emp = get_midpoint(points[ep1_ix], points[ep2_ix])
                extra_points.append(emp)
            midpoints.append(points_idx)
        triangle = np.append(con, midpoints)
        triangles.append(triangle)

    new_triangles = np.array(triangles)
    new_points = np.append(points, extra_points, axis=0)

    mesh.triangles6 = new_triangles
    mesh.points = new_points


def get_midpoint(p1, p2):
    # return [0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])]
    return 0.5 * (p1 + p2)


def solve_poisson(meshfile, param):
    mesh = MeshOps(meshfile)

    if param["order"] != 1:
        pass
    add_triangle6from3(mesh)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # x = mesh.points[:, 0]
    # y = mesh.points[:, 1]
    # ax.plot_trisurf(x, y, np.zeros_like(x), triangles=mesh.triangles, cmap="viridis")
    # ax.scatter(x, y, 0)
    # plt.show()
    print_mat = False

    A, f = assemble_poisson(mesh, param)
    A, f = apply_bc_poisson(mesh, A, f, param)
    if print_mat:
        print(A.toarray())
        print(f)
    un = sp.sparse.linalg.spsolve(A.tocsr(), f)

    if print_mat:
        print(un)

    # visualize on values on the nodes
    # ( accurate for S1 and S2 approximate space)

    N = len(un)
    vertxs = mesh.getNumberNodes()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    x6 = mesh.points[:N, 0]
    y6 = mesh.points[:N, 1]
    x = mesh.points[:vertxs, 0]
    y = mesh.points[:vertxs, 1]

    z = un if param["order"] == 1 else un[:vertxs]
    ax.plot_trisurf(x, y, z, triangles=mesh.triangles, cmap="viridis")
    ax.scatter(x6, y6, un)

    plt.show()


def calc_all_res_points(mesh, un):
    result = np.zeros_like(mesh.points[:, 0])
    for e in range(mesh.getNumberOfTriangles()):

        con = mesh.getNodeNumbersOfTriangle(e)


param_poisson = dict(
    laplaceCoeff=1,
    source=lambda x, y: 1,
    # source=lambda x, y: np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),
    dirichlet=0,
    neumann=1,
    order=2,
)
# solve_poisson("mesh/unitSquareStokes.msh", param_poisson)
# solve_poisson("mesh/unitSquare2.msh", param_poisson)
solve_poisson("mesh/unitSquare1.msh", param_poisson)
