import cv2
import numpy as np


def low_pass_filter_demo(image_name):
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    img_float32 = np.float32(image)

    rows, cols = image.shape
    crow, ccol = rows//2 , cols//2

    # FFT变换
    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 创建低通滤波器，低频区域为 1， 高频区域为 0
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    # 滤波
    fshift = dft_shift*mask

    # 逆变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    cv2.normalize(img_back, img_back, 0, 1.0, cv2.NORM_MINMAX)

    cv2.imshow("input", image)
    cv2.imshow("low-pass-filter", img_back)
    cv2.imwrite("./low_pass.png", np.uint8(img_back*255))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def high_pass_filter_demo(image_name):
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    img_float32 = np.float32(image)

    rows, cols = image.shape
    crow, ccol = rows//2 , cols//2

    # FFT变换
    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 创建高通滤波器，低频区域为 0， 高频区域为 1
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 0

    # 滤波
    fshift = dft_shift*mask

    # 逆变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    cv2.normalize(img_back, img_back, 0, 1.0, cv2.NORM_MINMAX)

    cv2.imshow("input", image);
    cv2.imshow("high-pass-filter", img_back)
    cv2.imwrite("./high_pass.png", np.uint8(img_back*255))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    image_path_6_1 = './image_test/1_1.jpg'
    image_path_6_2 = './image_test/1_2.jpg'
    image_path_7_1 = './image_test/2_1.jpg'
    image_path_7_2 = './image_test/2_2.jpg'
    image_path_8_1 = './image_test/3_1.jpg'
    image_path_8_2 = './image_test/3_2.jpg'
    image_path_9_1 = './image_test/4_1.jpg'
    image_path_9_2 = './image_test/4_2.jpg'
    
    # low_pass_filter_demo(image_path_6_1)
    high_pass_filter_demo(image_path_6_1)