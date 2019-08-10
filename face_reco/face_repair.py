import dlib

import cv2


def face_detector(img):
    # 人脸分类器,装饰器
    detector = dlib.get_frontal_face_detector()
    
    # img = cv2.imread(image_path)
    
    # 识别结果
    face_results = detector(img, 1)
    print("Number of faces detected: {}".format(len(face_results)))
    
    for face_result in face_results:
        # 绘制人脸区域
        cv2.rectangle(img, (face_result.left(), face_result.top()), (face_result.right(), face_result.bottom()),
                      (0, 0, 255), 1)
    
    cv2.imshow('face_results', img)
    cv2.waitKey(0)
    
    
if __name__ == '__main__':
    image_path_6_1 = './image_test/1_1.jpg'
    image_path_6_2 = './image_test/1_2.jpg'
    image_path_7_1 = './image_test/2_1.jpg'
    image_path_7_2 = './image_test/2_2.jpg'
    image_path_8_1 = './image_test/3_1.jpg'
    image_path_8_2 = './image_test/3_2.jpg'
    image_path_9_1 = './image_test/4_1.jpg'
    image_path_9_2 = './image_test/4_2.jpg'
    
    img = cv2.imread(image_path_8_1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, bin_pic = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    allMask = cv2.adaptiveThreshold(bin_pic, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mask = cv2.morphologyEx(allMask, cv2.MORPH_OPEN, kernel)
    
    cv2.imshow('bin', bin_pic)
    
    cv2.imshow('mask', mask)
    
    # mask为修复掩膜，为8位单通道图像，其中非零像素表示要修补的区域
    img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    
    cv2.imwrite('test.jpg', img)
    face_detector(img)