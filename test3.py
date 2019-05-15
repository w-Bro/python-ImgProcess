
"""
1. 图像通道处理
"""
import cv2

if __name__ == '__main__':
    
    img = cv2.imread('test.jpg', cv2.IMREAD_UNCHANGED)

    # ROI截取
    cv2.imshow('ROI_TEST', img[0: 100, 0: 100])
    
    b, g, r = cv2.split(img)
    
    cv2.imshow('blue', b)
    cv2.imshow('green', g)
    cv2.imshow('red', r)
    
    img = cv2.merge([r, g, b])
    cv2.imshow('merge', img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()