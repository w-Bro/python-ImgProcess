
import cv2
import os

MASK_PATH = '../id_card_mask'

"""
SIFT进行特征点（关键点）提取示例
"""

if __name__ == '__main__':
    
    mask_img_name = 'idcard_back_mask.jpg'
    mask_img_path = os.path.join(MASK_PATH, mask_img_name)
    
    # 0 -- IMREAD_GRAYSCALE 以灰度模式读入图像
    mask_img_gray = cv2.imread(mask_img_path, 0)
    
    cv2.namedWindow('gray', 0)
    cv2.imshow('gray', mask_img_gray)
    cv2.waitKey(0)
    
    try:
        # 使用SIFT进行特征点提取,构造一个sift对象
        sift = cv2.xfeatures2d.SIFT_create()
        
        # 测关键点和对应的描述子 kp是关键点列表，des是形状为Number_of_Keypoints×128的numpy数组,描述关键点的向量
        kp_mask, des_mask = sift.detectAndCompute(mask_img_gray, None)
        
        print(kp_mask, len(kp_mask))
        print(des_mask, len(des_mask[0]))
        
        # kp_img, des_img = sift.detectAndCompute(img_gray, None)

        # 在图像上绘制关键点
        queryImage = cv2.drawKeypoints(image=mask_img_gray, keypoints=kp_mask, outImage=None, color=(255, 0, 255),
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # trainImage = cv2.drawKeypoints(image=img_gray, keypoints=kp_img, outImage=None, color=(255, 0, 255),
        
        cv2.namedWindow('queryImage', 0)
        cv2.imshow('queryImage', queryImage)
        cv2.waitKey(0)
        
    except Exception as e:
        print(e)