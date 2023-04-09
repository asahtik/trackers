import time
from typing import Tuple
import numpy as np
import cv2

from matplotlib import pyplot as plt

from utils import gaussderiv, gausssmooth, show_flow, rotate_image

SIGMA_PERCENTAGE = 0.01
NEIGHOURHOOD_PERCENTAGE = 0.04
DIFF_EPSILON = 1e-8

SEED = 42

EPSILON = 1e-8

Output = Tuple[np.ndarray, np.ndarray]

def time_f(*args):
    def _time_f(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            out = func(*args, **kwargs)
            end = time.time()
            print(f'{func.__name__} took {end - start} seconds.')
            return out
        return wrapper
    if len(args) == 1 and callable(args[0]):
        return _time_f(args[0])
    return _time_f

def debug_show_img(img: np.ndarray):
    cv2.imshow("debug", img)
    cv2.waitKey(0)

def get_gauss_kernel(size: int, sigma: float) -> np.ndarray:
    size += 1 - size % 2
    return cv2.getGaussianKernel(size, sigma)

def get_image(filename) -> np.ndarray:
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'Error: could not read image {filename}.')
        exit(-1)
    img = img.astype(np.float32) / 255
    return img

def get_deriv(im1: np.ndarray, im2: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    Ix1, Iy1 = gaussderiv(im1, sigma)
    Ix2, Iy2 = gaussderiv(im2, sigma)
    Ix = (Ix1 + Ix2) / 2
    Iy = (Iy1 + Iy2) / 2
    return Ix, Iy

def harris(im: np.ndarray, N: int, k: float, sigma: float, gauss_kernel: bool = True) -> np.ndarray:
    # im − image matrix (grayscale)
    # N − size of the neighborhood (N × N)
    # k − Harris parameter
    N += 1 - N % 2

    Ix, Iy = gaussderiv(im, sigma)

    kernel = np.ones((N, 1))
    if gauss_kernel:
        kernel = get_gauss_kernel(N, N / 6)

    Ix2 = cv2.sepFilter2D(Ix * Ix, -1, kernel, kernel.T)
    Iy2 = cv2.sepFilter2D(Iy * Iy, -1, kernel, kernel.T)
    IxIy = cv2.sepFilter2D(Ix * Iy, -1, kernel, kernel.T)

    return Ix2 * Iy2 - IxIy * IxIy - k * (Ix2 + Iy2)**2

def harris_from_deriv(Ix: np.ndarray, Iy: np.ndarray, N: int, k: float, gauss_kernel: bool = True) -> np.ndarray:
    N += 1 - N % 2
    kernel = np.ones((N, 1))
    if gauss_kernel:
        kernel = get_gauss_kernel(N, N / 6)

    Ix2 = cv2.sepFilter2D(Ix * Ix, -1, kernel, kernel.T)
    Iy2 = cv2.sepFilter2D(Iy * Iy, -1, kernel, kernel.T)
    IxIy = cv2.sepFilter2D(Ix * Iy, -1, kernel, kernel.T)

    return Ix2 * Iy2 - IxIy * IxIy - k * (Ix2 + Iy2)**2

@time_f
def lucaskanade(im1: np.ndarray, im2: np.ndarray, N: int, sigma: float, time_smooth: bool = False, harris: Tuple[int, float, bool] | None = None) -> Output:
    # im1 − first image matrix (grayscale)
    # im2 − second image matrix (grayscale)
    # N − size of the neighborhood (N × N)
    It = im2 - im1 # type: ignore
    It = gausssmooth(It, sigma)
    Ix, Iy = get_deriv(im1, im2, sigma)

    harris_response_mask = None
    if harris:
        alpha = 0.001
        harris_response = harris_from_deriv(Ix, Iy, harris[0], harris[1], harris[2])
        harris_response = harris_response / np.max(harris_response)
        harris_response[harris_response < alpha] = 0.0
        harris_response = gausssmooth(harris_response, sigma)
        harris_response = harris_response / np.max(harris_response)
        harris_response_mask = (harris_response > alpha).astype(np.float32)

    Ix2 = cv2.sepFilter2D(Ix * Ix, -1, np.ones((N, 1)), np.ones((1, N)))
    Iy2 = cv2.sepFilter2D(Iy * Iy, -1, np.ones((N, 1)), np.ones((1, N)))
    IxIy = cv2.sepFilter2D(Ix * Iy, -1, np.ones((N, 1)), np.ones((1, N)))
    IxIt = cv2.sepFilter2D(Ix * It, -1, np.ones((N, 1)), np.ones((1, N)))
    IyIt = cv2.sepFilter2D(Iy * It, -1, np.ones((N, 1)), np.ones((1, N)))

    D: np.ndarray = Ix2 * Iy2 - IxIy**2

    D += (D < EPSILON).astype(np.float32)
    mask = (D >= EPSILON).astype(np.float32)

    u = ((-(Iy2 * IxIt - IxIy * IyIt)) / D) * mask
    v = ((-(Ix2 * IyIt - IxIy * IxIt)) / D) * mask

    if harris_response_mask is not None:
        u *= harris_response_mask
        v *= harris_response_mask

    return u, v

@time_f
def hornschunk(im1: np.ndarray, im2: np.ndarray, n_iters: int, lmbd: float, sigma: float, time_smooth: bool = False, uv_init: Output | None = None) -> Output:
    # im1 − first image matrix (grayscale)
    # im2 − second image matrix (grayscale)
    # n_iters - number of iterations
    # lmbd - parameter
    avg_kernel = np.array([
        [0, 1/4, 0],
        [1/4, 0, 1/4],
        [0, 1/4, 0]
    ]) # type: ignore

    It = im2 - im1 # type: ignore
    if time_smooth:
        It = gausssmooth(It, sigma)
    Ix, Iy = get_deriv(im1, im2, sigma)
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy

    D = Ix2 + Iy2 + lmbd

    if uv_init is None:
        u1 = np.zeros_like(im1)
        v1 = np.zeros_like(im1)
    else:
        u1, v1 = uv_init
    u2 = u1.copy()
    v2 = v1.copy()
    iter_end = 0
    for i in range(0, n_iters):
        iter_end = i
        P = (It + Ix * u1 + Iy * v1) / D # type: ignore
        u2 = u1 - Ix * P
        v2 = v1 - Iy * P

        u_sim = u1.flatten().dot(u2.flatten()) / (np.linalg.norm(u1) * np.linalg.norm(u2) + 1e-8)
        v_sim = v1.flatten().dot(v2.flatten()) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        if 1 - abs(u_sim) < DIFF_EPSILON and 1 - abs(v_sim) < DIFF_EPSILON and i > min(100, n_iters // 2):
            break
        u1 = cv2.filter2D(u2, -1, avg_kernel)
        v1 = cv2.filter2D(v2, -1, avg_kernel)
    print(iter_end)
    return u2, v2

def show_flow_overlay(u, v, image):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    show_flow(u, v, ax, type="field", set_aspect=True)
    extent = (0, image.shape[1], -image.shape[0], 0)
    plt.imshow(image, alpha=0.7, extent=extent)
    plt.axis('off')
    plt.tight_layout()

def show_flow_overlay2(u, v, image, ax):
    show_flow(u, v, ax, type="field", set_aspect=True)
    extent = (0, image.shape[1], -image.shape[0], 0)
    ax.imshow(image, alpha=0.7, cmap="gray", extent=extent)

def show_flow2(im1: np.ndarray, im2: np.ndarray, U_lk: np.ndarray, V_lk: np.ndarray, U_hs: np.ndarray, V_hs: np.ndarray, name: str):
    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1)
    ax1_12.imshow(im2)
    show_flow(U_lk, V_lk, ax1_21, type="angle")
    show_flow(U_lk, V_lk, ax1_22, type="field", set_aspect=True)
    fig1.suptitle("Lucas-Kanade Optical Flow")
    fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
    ax2_11.imshow(im1)
    ax2_12.imshow(im2)
    show_flow(U_hs, V_hs, ax2_21, type="angle")
    show_flow(U_hs, V_hs, ax2_22, type="field", set_aspect=True)
    fig2.suptitle("Horn-Schunck Optical Flow")
    fig1.savefig(f"LK_{name}.png")
    fig2.savefig(f"HS_{name}.png")
    plt.show()

def show_flow3(im1: np.ndarray, im2: np.ndarray, U_lk: np.ndarray, V_lk: np.ndarray, U_hs: np.ndarray, V_hs: np.ndarray, name: str):
    fig, ((ax_11, ax_12), (ax_21, ax_22), (ax_31, ax_32)) = plt.subplots(3, 2)
    ax_11.imshow(im1)
    ax_12.imshow(im2)
    show_flow(U_lk, V_lk, ax_21, type="angle")
    ax_21.set_title("Lucas-Kanade Optical Flow")
    show_flow(U_hs, V_hs, ax_22, type="angle")
    ax_22.set_title("Horn-Schunck Optical Flow")
    show_flow(U_lk, V_lk, ax_31, type="field", set_aspect=True)
    show_flow(U_hs, V_hs, ax_32, type="field", set_aspect=True)
    fig.tight_layout()
    ax_11.axis('off')
    ax_12.axis('off')
    ax_21.axis('off')
    ax_22.axis('off')
    ax_31.axis('off')
    ax_32.axis('off')
    fig.savefig(f"OF_{name}.png", dpi=300)
    plt.show()

def test_params_lk():
    im1 = get_image("collision/00000150.jpg")
    im2 = get_image("collision/00000151.jpg")

    U_1, V_1 = lucaskanade(im1, im2, 1, 1.0, time_smooth=True)
    U_2, V_2 = lucaskanade(im1, im2, 20, 1.0, time_smooth=True)
    U_3, V_3 = lucaskanade(im1, im2, 200, 1.0, time_smooth=True)

    U_4, V_4 = lucaskanade(im1, im2, 20, 0.5, time_smooth=True)
    U_5, V_5 = lucaskanade(im1, im2, 20, 2.0, time_smooth=True)
    U_6, V_6 = lucaskanade(im1, im2, 20, 20, time_smooth=True)

    fig, ((ax_11, ax_12), (ax_21, ax_22), (ax_31, ax_32)) = plt.subplots(3, 2)
    show_flow_overlay2(U_1, V_1, im1, ax_11)
    show_flow_overlay2(U_2, V_2, im1, ax_21)
    show_flow_overlay2(U_3, V_3, im1, ax_31)
    show_flow_overlay2(U_4, V_4, im1, ax_12)
    show_flow_overlay2(U_5, V_5, im1, ax_22)
    show_flow_overlay2(U_6, V_6, im1, ax_32)
    ax_11.set_title("Neighborhood size")
    ax_12.set_title("Sigma")
    fig.suptitle("Parameter comparisons")
    fig.tight_layout()
    ax_11.axis('off')
    ax_12.axis('off')
    ax_21.axis('off')
    ax_22.axis('off')
    ax_31.axis('off')
    ax_32.axis('off')
    fig.savefig("LK_params.png", dpi=300)
    plt.show()

def test_params_hs():
    im1 = get_image("collision/00000150.jpg")
    im2 = get_image("collision/00000151.jpg")

    U_1, V_1 = hornschunk(im1, im2, 100, 0.5, time_smooth=True, sigma=2.0)
    U_2, V_2 = hornschunk(im1, im2, 1000, 0.5, time_smooth=True, sigma=2.0)
    U_3, V_3 = hornschunk(im1, im2, 10000, 0.5, time_smooth=True, sigma=2.0)

    U_4, V_4 = hornschunk(im1, im2, 5000, 0.1, time_smooth=True, sigma=2.0)
    U_5, V_5 = hornschunk(im1, im2, 5000, 0.5, time_smooth=True, sigma=2.0)
    U_6, V_6 = hornschunk(im1, im2, 5000, 1.0, time_smooth=True, sigma=2.0)

    fig, ((ax_11, ax_12), (ax_21, ax_22), (ax_31, ax_32)) = plt.subplots(3, 2)
    show_flow_overlay2(U_1, V_1, im1, ax_11)
    show_flow_overlay2(U_2, V_2, im1, ax_21)
    show_flow_overlay2(U_3, V_3, im1, ax_31)
    show_flow_overlay2(U_4, V_4, im1, ax_12)
    show_flow_overlay2(U_5, V_5, im1, ax_22)
    show_flow_overlay2(U_6, V_6, im1, ax_32)
    ax_11.set_title("Iterations")
    ax_12.set_title("k")
    fig.suptitle("Parameter comparisons")
    fig.tight_layout()
    ax_11.axis('off')
    ax_12.axis('off')
    ax_21.axis('off')
    ax_22.axis('off')
    ax_31.axis('off')
    ax_32.axis('off')
    fig.savefig("HS_params.png", dpi=300)
    plt.show()

def test_speedup():
    im3 = get_image("collision/00000150.jpg")
    im4 = get_image("collision/00000151.jpg")

    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2)

    im1 = np.random.rand(200, 200).astype(np.float32)
    im2 = im1.copy()
    im2 = rotate_image(im2, -1)

    U_lk, V_lk = lucaskanade(im1, im2, 3, 1.0, time_smooth=True)
    U_hs, V_hs = hornschunk(im1, im2, 10, 0.5, 1.0, time_smooth=True)
    U_hss, V_hss = hornschunk(im1, im2, 10, 0.5, 1.0, time_smooth=True, uv_init=(U_lk, V_lk))
    show_flow_overlay2(U_hs, V_hs, np.zeros_like(im1), ax11)
    show_flow_overlay2(U_hss, V_hss, np.zeros_like(im1), ax12)

    U_lk, V_lk = lucaskanade(im3, im4, 25, 2.0, time_smooth=True)
    U_hs, V_hs = hornschunk(im3, im4, 500, 0.5, 2.0, time_smooth=True)
    U_hss, V_hss = hornschunk(im3, im4, 500, 0.5, 2.0, time_smooth=True, uv_init=(U_lk, V_lk))
    show_flow_overlay2(U_hs, V_hs, im3, ax21)
    show_flow_overlay2(U_hss, V_hss, im3, ax22)

    ax11.axis('off')
    ax12.axis('off')
    ax21.axis('off')
    ax22.axis('off')

    fig.suptitle("Initialized Horn-Schunck convergence")
    fig.tight_layout()
    fig.savefig("HS_speedup.png", dpi=300)
    plt.show()

def test():
    np.random.seed(SEED)
    im1 = np.random.rand(200, 200).astype(np.float32)
    im2 = im1.copy()
    im2 = rotate_image(im2, -1)
    U_lk, V_lk = lucaskanade(im1, im2, 3, 1.0, time_smooth=True)
    U_hs, V_hs = hornschunk(im1, im2, 1000, 0.5, 1.0)
    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1)
    ax1_12.imshow(im2)
    show_flow(U_lk, V_lk, ax1_21, type="angle")
    show_flow(U_lk, V_lk, ax1_22, type="field", set_aspect=True)
    fig1.suptitle("Lucas-Kanade Optical Flow")
    fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
    ax2_11.imshow(im1)
    ax2_12.imshow(im2)
    show_flow(U_hs, V_hs, ax2_21, type="angle")
    show_flow(U_hs, V_hs, ax2_22, type="field", set_aspect=True)
    fig2.suptitle("Horn-Schunck Optical Flow")
    fig1.savefig("LK_bench.png")
    fig2.savefig("HS_bench.png")
    plt.show()

def test2():
    im1 = get_image("collision/00000150.jpg")
    im2 = get_image("collision/00000151.jpg")

    size = min(im1.shape[0], im1.shape[1])

    neighourhood = int(size * NEIGHOURHOOD_PERCENTAGE)
    sigma = size * SIGMA_PERCENTAGE

    U_nh, V_nh = lucaskanade(im1, im2, neighourhood, sigma, time_smooth=True)
    U_hs, V_hs = lucaskanade(im1, im2, neighourhood, sigma, time_smooth=True, harris=(neighourhood, 0.04, True))


    fig, ((ax_11, ax_12), (ax_21, ax_22), (ax_31, ax_32)) = plt.subplots(3, 2)
    ax_11.imshow(im1)
    ax_12.imshow(im2)
    show_flow(U_nh, V_nh, ax_21, type="angle")
    ax_21.set_title("Lucas-Kanade without Harris")
    show_flow(U_hs, V_hs, ax_22, type="angle")
    ax_22.set_title("Lucas-Kanade with Harris")
    show_flow(U_nh, V_nh, ax_31, type="field", set_aspect=True)
    show_flow(U_hs, V_hs, ax_32, type="field", set_aspect=True)
    fig.tight_layout()
    ax_11.axis('off')
    ax_12.axis('off')
    ax_21.axis('off')
    ax_22.axis('off')
    ax_31.axis('off')
    ax_32.axis('off')
    fig.savefig(f"OF_harris.png", dpi=300)
    plt.show()

def main():
    im1 = get_image("disparity/cporta_left.png")
    im2 = get_image("disparity/cporta_right.png")

    size = min(im1.shape[0], im1.shape[1])

    neighourhood = int(size * NEIGHOURHOOD_PERCENTAGE)
    sigma = size * SIGMA_PERCENTAGE

    u_lk, v_lk = lucaskanade(im1, im2, neighourhood, sigma, time_smooth=True)
    u_hs, v_hs = hornschunk(im1, im2, 10000, 0.5, time_smooth=True, sigma=sigma)

    show_flow3(im1, im2, u_lk, v_lk, u_hs, v_hs, "collision")

if __name__ == '__main__':
    test()