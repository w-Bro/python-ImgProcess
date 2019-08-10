
import pytesseract
import cv2
"""
OCR示例
    -- 固定阈值二值化
    -- 自适应阈值二值化
"""
if __name__ == '__main__':
    
    # 灰度图
    img_gray = cv2.imread('name.jpg', 0)
    
    # 自适应阈值二值化
    binary_pic = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 40)
    
    cv2.namedWindow('bin1', 0)
    cv2.imshow('bin1', binary_pic)
    cv2.waitKey(0)

    # psm:
    # 0：定向脚本监测（OSD）
    # 1： 使用OSD自动分页
    # 2 ：自动分页，但是不使用OSD或OCR（Optical Character Recognition，光学字符识别）
    # 3 ：全自动分页，但是没有使用OSD（默认）
    # 4 ：假设可变大小的一个文本列。
    # 5 ：假设垂直对齐文本的单个统一块。
    # 6 ：假设一个统一的文本块。
    # 7 ：将图像视为单个文本行。
    # 8 ：将图像视为单个词。
    # 9 ：将图像视为圆中的单个词。
    # 10 ：将图像视为单个字符
    s = pytesseract.image_to_string(binary_pic, lang='chi_sim', config=f"-psm 7").replace(' ', '').replace('\n', '')
    print('自适应阈值二值化OCR结果：', s)

    # 二值化
    (_, binary_pic) = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    cv2.namedWindow('bin2', 0)
    cv2.imshow('bin2', binary_pic)
    cv2.waitKey(0)
    s = pytesseract.image_to_string(binary_pic, lang='chi_sim', config=f"-psm 7").replace(' ', '').replace('\n', '')
    print('固定阈值二值化OCR结果：', s)