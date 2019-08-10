import cv2
from find_id_card import img_resize_gray
"""
模板匹配示例
"""

if __name__ == '__main__':
    
    # 模板图
    template_path = '../id_card_mask/name_mask_1280.jpg'
    template = cv2.UMat(cv2.imread(template_path, 0))
    w, h = cv2.UMat.get(template).shape[::-1]
    
    # 目标图
    img = cv2.UMat(cv2.imread('demo2.jpg'))
    img_gray, img_org = img_resize_gray(img)
    
    # 模板匹配
    # - CV_TM_SQDIFF 平方差匹配法：该方法采用平方差来进行匹配；最好的匹配值为0；匹配越差，匹配值越大。
    # - CV_TM_SQDIFF_NORMED 相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好。
    # - CV_TM_CCORR 相关系数匹配法：1 表示完美的匹配；-1 表示最差的匹配。
    # - CV_TM_CCORR_NORMED 归一化平方差匹配法
    # - CV_TM_CCOEFF 归一化相关匹配法
    # - CV_TM_CCOEFF_NORMED 归一化相关系数匹配法
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    cv2.namedWindow('result', 0)
    cv2.imshow('result', res)
    cv2.waitKey(0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    x = 1280.00 / 3840.00
    
    # 估算目标区域位置
    top_left = (max_loc[0] + w, max_loc[1] - int(20 * x))
    bottom_right = (top_left[0] + int(700 * x), top_left[1] + int(300 * x))
    
    result = cv2.UMat.get(img_org)[top_left[1] - 10: bottom_right[1], top_left[0] - 10: bottom_right[0]]
    
    # 绘制模板识别区域
    cv2.rectangle(img_org, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
    # 绘制估算区域
    cv2.rectangle(img_org, top_left, bottom_right, (255, 0, 0), 2)
    
    cv2.namedWindow('img_org', 0)
    cv2.imshow('img_org', img_org)
    cv2.waitKey(0)
    
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.imwrite('demo3.jpg', result)
    # cv2.namedWindow('name', 0)
    # cv2.imshow('name', result)
    # cv2.waitKey(0)