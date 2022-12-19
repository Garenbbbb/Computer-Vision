import pathlib

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy import interpolate
import cv2

DIR = 'output/part1/'
pathlib.Path(DIR).mkdir(parents=True, exist_ok=True)


def read_inputs():
    """
    :return: (101, h, w) images, (2*101, N) matrix

    """
    ret_im = []
    for i in range(1, 102):
        ret_im.append(cv2.imread(
            'factorization_data/frame%s.jpg' % (str(i).zfill(8)),
            cv2.IMREAD_GRAYSCALE
        ))
    mtx = np.loadtxt('factorization_data/measurement_matrix.txt')
    return np.array(ret_im), mtx


def normalize(data_mtx):
    mean = np.mean(data_mtx, axis=1, keepdims=True)
    data_mtx = data_mtx - mean
    return data_mtx, mean


def solve_M_S(data_mtx):
    U, D, Vt = la.svd(data_mtx)
    D = np.diag(np.sqrt(D[:3]))
    M = U[:, :3] @ D
    S = D @ Vt[:3]
    # find data_mtx = M @ Q @ Q.inv() @ S
    equations = []
    values = []
    m = data_mtx.shape[0] // 2
    for i in range(m):
        Ai = M[2 * i:2 * i + 2]
        # find L s.t. Ai @ L @ Ai.T = I
        equations.append(np.outer(Ai[0], Ai[0]).reshape(-1))
        values.append(1)
        equations.append(np.outer(Ai[0], Ai[1]).reshape(-1))
        values.append(0)
        equations.append(np.outer(Ai[1], Ai[0]).reshape(-1))
        values.append(0)
        equations.append(np.outer(Ai[1], Ai[1]).reshape(-1))
        values.append(1)
    L = la.lstsq(np.array(equations),
                 np.array(values), rcond=None)[0].reshape((3, 3))
    Q = la.cholesky(L)
    print(Q)
    M = M @ Q
    S = la.pinv(Q) @ S
    return M, S


def part1_pipeline():
    imgs, raw_data_mtx = read_inputs()
    data_mtx, means = normalize(raw_data_mtx)
    P_matrices, pts3 = solve_M_S(data_mtx)

    # ax = plt.axes(projection='3d')
    # ax.view_init(45, 90)
    # ax.scatter(pts3[0], pts3[1], pts3[2],
    #         c='b', label='3D points')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # #ax.view_init(elev=-97, azim=160)
    # # ax.view_init(elev=190, azim=-20, vertical_axis='x')
    # plt.legend()
    # plt.show()
    # proj2d = (P_matrices @ pts3) + means
    # # selected_frames = [1, 10, 20]
    for f in range(len(imgs)):
        a = np.linalg.norm(raw_data_mtx[f] - pts3[f]) ** 2
    # #     plt.figure()
    # #     fig = plt.figure()
    # #     ax = fig.add_subplot(1, 1, 1, projection='3d')
    # #     ax.imshow(imgs[f], cmap='gray')
    # #     # ax.scatter(raw_data_mtx[2 * f], raw_data_mtx[2 * f + 1], c='g',
    # #     #             label='Observed feature points')
    # #     ax.scatter(proj2d[2 * f], proj2d[2 * f + 1], c='r', marker='+',
    # #                 label='Estimated feature points')
        

    # #     ax.view_init(60, 90)
    # #     ax.set_xlabel('x')
    # #     ax.set_ylabel('y')
    # #     ax.set_zlabel('z')
    # proj2d = (P_matrices @ pts3) + means
    # res = 0
    # selected_frames = [50]
    # x = []
    # y = []
    # print(len(imgs))
    # for f in range(len(imgs)//2):
    #     #a = sum((raw_data_mtx[2*f] - proj2d[2*f])**2)
    #     a = np.linalg.norm(raw_data_mtx[2*f] - proj2d[2*f]) ** 2
    #     # b = (raw_data_mtx[2 * f + 1] - proj2d[2 * f + 1]) ** 2
    #     # r = a+b
    #     res += a
    #     x.append(f*2)
    #     # x.append(f)
    #     y.append(a)
    #     # plt.figure()
    #     # plt.imshow(imgs[f], cmap='gray')
    #     # plt.scatter(raw_data_mtx[2 * f], raw_data_mtx[2 * f + 1], c='g',
    #     #             label='Observed feature points')
    #     # plt.scatter(proj2d[2 * f], proj2d[2 * f + 1], c='r', marker='+',
    #     #             label='Estimated feature points')

    # #plt.scatter(x, y)
    # plt.plot(x, y)
    # plt.show()
    # print(res)  
    # x_new = np.linspace(1, 120)
    # bspline = interpolate.make_interp_spline(x, y)
    # y_new = bspline(x_new)

    # plt.plot(x_new, y_new)

    # plt.show()
    # plt.grid()
    # plt.plot(x, y)#, marker="o")#, markersize=1, markeredgecolor="red", markerfacecolor="green")
    # plt.show()  
   




part1_pipeline()
