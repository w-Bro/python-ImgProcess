import cv2
import os
import numpy as np

MASK_PATH = '../id_card_mask'

"""
使用FLANN算法进行特征点（关键点）匹配，计算得出身份证区域
"""

if __name__ == '__main__':
    
    # 原图
    mask_img_name = 'idcard_back_mask.jpg'
    # 目标图
    img_name = '0.jpg'
    mask_img_path = os.path.join(MASK_PATH, mask_img_name)
    img_org_path = os.path.join('../image_test', img_name)
    
    # 0 -- IMREAD_GRAYSCALE 以灰度模式读入图像
    mask_img_gray = cv2.UMat(cv2.imread(mask_img_path, 0))
    
    img_org = cv2.imread(img_org_path)
    img_gray = cv2.UMat(cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY))
    
    try:
        # 使用SIFT进行特征点提取,构造一个sift对象
        sift = cv2.xfeatures2d.SIFT_create()
        # 测关键点和对应的描述子 kp是关键点列表，des是形状为Number_of_Keypoints×128的numpy数组,描述关键点的向量
        kp_mask, des_mask = sift.detectAndCompute(mask_img_gray, None)
        kp_img, des_img = sift.detectAndCompute(img_gray, None)
        
        # 参数 -- 指定算法 FLANN是快速估计最近邻的库。它包含了一些为大数据集内搜索快速近邻和高维特征的优化算法
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # 参数 -- 指定了索引里的树应该被递归遍历的次数。更高的值带来更高的准确率。但是也花更多时间
        search_params = dict(checks=10)
        # 基于FLANN的匹配器
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_mask, des_img, k=2)
        
        # 最少匹配特征点数量 10 改为 15
        MIN_MATCH_COUNT = 10
        
        img_org = cv2.UMat(img_org)
        
        # 两个最佳匹配之间距离越小越严格,采用关键点特征向量的欧式距离来作为两幅图像中关键点的相似性判定度量
        goods = []
        for m, n in matches:
            # 原0.7 改为 0.6
            if m.distance < 0.65 * n.distance:
                goods.append(m)
        
        if len(goods) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp_mask[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_img[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)
            
            # 用HomoGraphy计算图像与图像之间映射关系, M为转换矩阵，为求得的单应性矩阵, mask是一个列表来表征匹配成功的特征点
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            print('单应性矩阵:', M)
            # 将多维数组转换为一维数组
            matchesMask = mask.ravel().tolist()
            
            # 绘制匹配的关键点
            draw_params = dict(matchColor=(255, 0, 0), singlePointColor=None, matchesMask=None, flags=2)
            match_img = cv2.drawMatches(mask_img_gray, kp_mask, img_gray, kp_img, goods, outImg=None, **draw_params)
            cv2.namedWindow('match_img', 0)
            cv2.imshow('match_img', match_img)
            cv2.waitKey(0)
            
            # 原图像的宽高
            h, w = cv2.UMat.get(mask_img_gray).shape

            # 计算逆矩阵
            M_r = np.linalg.inv(M)
            # 实现透视变换转换,
            result = cv2.warpPerspective(img_org, M_r, (w, h))
            # 保存
            cv2.imwrite('demo2.jpg', result)
            
            # 身份证区域在原图中的坐标
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            print(dst)
            # 绘制出身份证区域
            target_img = cv2.polylines(img_org, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            cv2.namedWindow('target_img', 0)
            cv2.imshow('target_img', img_org)
            cv2.waitKey(0)
            # 将识别到的身份证区域填充成黑色，以便识别下一张身份证
            # img = cv2.fillPoly(img_org, [np.int32(dst)], (0, 0, 0))

            cv2.namedWindow('result', 0)
            cv2.imshow('result', result)
            cv2.waitKey(0)
            
        else:
            print('匹配的特征点个数过少，无法确认为身份证')
    except Exception as e:
        print(str(e))
        pass
