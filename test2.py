
"""
1. 绘制自定义图像
"""

import cv2
import numpy as np

if __name__ == '__main__':
    # 依据给定形状和类型(shape[, dtype, order])返回一个新的元素全部为1的数组
    white_img = np.ones((512, 512, 3), np.uint8)
    # 每个像素点变为[255, 255, 255]
    white_img = 255 * white_img
    
    # 绘制直线
    cv2.line(white_img, (0, 0), (100, 100), (255, 0, 0), 2)

    # 绘制矩形
    cv2.rectangle(white_img, (100, 0), (200, 100), (0, 255, 0), 2)

    # 绘制实心圆形
    cv2.circle(white_img, (250, 50), 50, (0, 0, 255), -1)

    # 绘制实心圆形
    cv2.circle(white_img, (350, 50), 50, (0, 0, 255), 3)

    # 绘制椭圆
    cv2.ellipse(white_img, (100, 150), (100, 50), 0, 0, 360, (0, 0, 255), 3)
    
    # 绘制多边形
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    cv2.polylines(white_img, [pts], True, (0, 0, 0), 2)

    cv2.imshow('white_img', white_img)
    
    # 对图片取反
    reversed_white_img = 255 - white_img
    cv2.imshow('reversed_white_img', reversed_white_img)
    # 接收指定的键盘敲击再关闭窗口 -- ESC
    if cv2.waitKey(0) == 27:
        # 销毁全部窗口
        cv2.destroyAllWindows()