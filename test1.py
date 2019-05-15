
"""
1. 读取、展示和存储图像
2. 图像色彩空间变换
"""


import cv2
import imutils

if __name__ == '__main__':
    # 读取图像 -- 彩色图像
    img1 = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
    # 读取图像 -- 灰度模式
    img2 = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
    # 读取图像 -- 保留原有的颜色通道
    img3 = cv2.imread('test.jpg', cv2.IMREAD_UNCHANGED)
    
    # 在窗口显示图像 -- 自动适应图像大小
    cv2.imshow('auto size', img1)

    # 在窗口显示图像 -- 通过imutils模块改变图像显示大小
    cv2.imshow('imutils size', imutils.resize(img2, 800))

    # 接收指定的键盘敲击再关闭窗口 -- ESC
    if cv2.waitKey(0) == 27:
        # 销毁全部窗口
        cv2.destroyAllWindows()
   
    # 图像色彩处理
    cvt_result = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # 将图像保存到本地
    cv2.imwrite('test_gray.jpg', img2)