import numpy as np
from scipy.sparse import lil_matrix
from mesh_ops import MeshOps
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt


def assemble_poisson(mesh: MeshOps, param):
    # TODO!
    N = mesh.getNumberNodes()
    A = lil_matrix((N, N))
    f = np.zeros(N)
    num_elements = mesh.getNumberOfTriangles()

    # Shape functions on P1 reference element, assuming counter clockwise from (0,0)
    phi = [lambda x, y: 1 - x - y, lambda x, y: x, lambda x, y: y]

    # points for quadrature on reference element from hw5
    pts = [[1 / 6, 1 / 6], [4 / 6, 1 / 6], [1 / 6, 4 / 6]]
    source = param["source"]

    # Iterate over all elements
    for e in range(num_elements):
        ########################
        # calculate contributions of every element to each integral for the
        # shape functions on the vertecies of the element  (for bilinear form / matrix)
        #########################

        elemA = np.zeros((3, 3))
        # gradient of shape functions on ref triangle
        dphi_ref = np.array([[-1, -1], [1, 0], [0, 1]])
        invJ = mesh.calcInverseJacobianOfTriangle(e)
        J = mesh.calcJacobianOfTriangle(e)
        # calculates (B_K ^-T grad phi_n)^T = grad phi_n^T B_K^-1
        # this is done all at once by stacking the transposed gradients on top of each other
        # to a 3x2 matrix resulting in stacked result row vectors in a 3x2 matrix
        dphi = dphi_ref @ invJ
        detJ = mesh.calcJacobianDeterminantOfTriangle(e)
        # (dphi @ dphi.T) will calculate a matrix with a_ij = phi_i * phi_j <- dot product
        # 1/2 because the integral is constant, pulling everything out leaves the area as a factor
        # Area of a triangle with sidelenghts 1 is 1/2
        elemA = (dphi @ dphi.T) * detJ * 1 / 2
        con = mesh.getNodeNumbersOfTriangle(e)
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
            for pt in pts:
                x = p1 + J @ pt
                temp += phi[i](pt[0], pt[1]) * source(pt[0], pt[1])
            f[node] += 1 / 6 * temp * detJ

    return A, f


def apply_bc_poisson(mesh: MeshOps, A, f, param):
    N = mesh.getNumberNodes()
    for i, node in enumerate(mesh.getNodeList()):
        x, y = node
        if x == 0 or y == 0 or y == 1:
            f[i] = 0
            A[i, :] = 0
            A[i, i] = 1
    return A, f


def solve_poisson(meshfile, param):
    mesh = MeshOps(meshfile)

    A, f = assemble_poisson(mesh, param)
    A, f = apply_bc_poisson(mesh, A, f, param)
    print(A.toarray())
    print(f)
    import scipy as sp

    un = sp.sparse.linalg.spsolve(A, f)

    print(un)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    ax.plot_trisurf(x, y, un, triangles=mesh.triangles)

    plt.show()

    # visualize on values on the nodes
    # ( accurate for S1 approximate space)


param_poisson = dict(
    laplaceCoeff=1,
    source=lambda x, y: 1,
    # source=lambda x, y: np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),
    dirichlet=0,
    neumann=1,
    order=1,
)
solve_poisson("mesh/unitSquare2.msh", param_poisson)
