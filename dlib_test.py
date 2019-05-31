
import cv2
import dlib


def face_detector(image_path):
    # 人脸分类器,装饰器
    detector = dlib.get_frontal_face_detector()
    
    img = cv2.imread(image_path)
    
    # 识别结果
    face_results = detector(img, 1)
    print("Number of faces detected: {}".format(len(face_results)))
    
    for face_result in face_results:
        # 绘制人脸区域
        cv2.rectangle(img, (face_result.left(), face_result.top()), (face_result.right(), face_result.bottom()),
                      (0, 0, 255), 1)
    
    cv2.imshow('face_results', img)
    cv2.waitKey(0)
    
    
# 人脸特征检测
def shape_predictor(image_path):
    # 模型
    model_path = './shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(model_path)

    # 人脸分类器,装饰器
    detector = dlib.get_frontal_face_detector()

    img = cv2.imread(image_path)

    # 识别结果
    face_results = detector(img, 1)
    
    for face_result in face_results:
        shape = predictor(img, face_result)
        for part in shape.parts():
            p = (part.x, part.y)
            cv2.circle(img, p, 2, (255, 0, 0), 1)
            
    cv2.imshow('predictor', img)
    cv2.waitKey(0)
    
    
# 调用训练好的卷积神经网络（CNN）模型进行人脸检测
# 用CPU运算会很慢
def face_detector_cnn(image_path):
    model_path = './mmod_human_face_detector.dat'
    # 导入cnn模型
    detecor = dlib.cnn_face_detection_model_v1(model_path)

    img = cv2.imread(image_path)
    
    face_results = detecor(img, 1)
    print("Number of faces detected: {}".format(len(face_results)))

    # 返回的结果是一个mmod_rectangles对象,包含有2个成员变量：
    # dlib.rectangle类，表示对象的位置；dlib.confidence，表示置信度
    for face_result in face_results:
        print(f"confidence: {face_result.confidence}")
        face = face_result.rect
        # 绘制人脸区域
        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()),
                      (0, 0, 255), 1)

    cv2.imshow('face_results_cnn', img)
    cv2.waitKey(0)
    
    
down_pos, selection, target, is_choose = None, None, None, False
# 鼠标绘制
def mouse_draw(event, x, y, flags, param):
    global down_pos, selection, target
    # 鼠标按下
    if event == cv2.EVENT_LBUTTONDOWN:
        down_pos = (x, y)
        print("鼠标按下")
    if down_pos is not None:
        xMin = min(x, down_pos[0])
        yMin = min(y, down_pos[1])
        xMax = max(x, down_pos[0])
        yMax = max(y, down_pos[1])
        selection = (xMin, yMin, xMax, yMax)
    if event == cv2.EVENT_LBUTTONUP:
        down_pos = None
        target = selection
        selection = None
        print("鼠标松开")
        print(f"targert: {target}")
        
        
# 单目标跟踪
def single_target_tracker():
            
    # 跟踪器
    tracker = dlib.correlation_tracker()
    # opencv打开摄像头
    cap = cv2.VideoCapture(0)

    cv2.namedWindow('single_target_tracker', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('single_target_tracker', mouse_draw)

    # 是否按下了空格并确认目标
    global is_choose
    """
    按下空格选择跟踪目标，再按下空格确认目标并开始跟踪
    """
    while True:
        ret, frame = cap.read()
        
        # 镜像翻转
        frame = cv2.flip(frame, 180)
        cv2.imshow('single_target_tracker', frame)
        
        if cv2.waitKey(5) == 32:
            is_choose = False
            # 选择目标
            print('请选择跟踪目标')
            while True:
                img = frame.copy()
                if target:
                    cv2.rectangle(img, (target[0], target[1]), (target[2], target[3]), (0, 0, 255), 1)
                elif selection:
                    cv2.rectangle(img, (selection[0], selection[1]), (selection[2], selection[3]), (0, 0, 255), 1)
                cv2.imshow("single_target_tracker", img)
                if cv2.waitKey(5) == 32:
                    is_choose = True
                    break
            # 跟踪目标
            print('目标选择完成')
            tracker.start_track(frame, dlib.rectangle(target[0], target[1], target[2], target[3]))

        if is_choose:
            tracker.update(frame)
        track_window = tracker.get_position()
        cv2.rectangle(frame, (int(track_window.left()), int(track_window.top())),
                      (int(track_window.right()), int(track_window.bottom())), (0, 255, 255), 1)
        cv2.imshow('single_target_tracker', frame)
        if cv2.waitKey(5) == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
    
# 人脸对比
# 将人脸的信息提取成一个128维的向量空间,度量采用欧式距离
def face_contrast(image_path_1, image_path_2):
    # 模型
    shape_model_path = './shape_predictor_68_face_landmarks.dat'
    face_recognition_model_path = './dlib_face_recognition_resnet_model_v1.dat'

    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(shape_model_path)
    face_rec_model = dlib.face_recognition_model_v1(face_recognition_model_path)
    
    image_1 = cv2.imread(image_path_1)
    image_2 = cv2.imread(image_path_2)
    
    # 人脸识别结果
    face_1_results = detector(image_1, 1)
    face_2_results = detector(image_2, 1)
    
    if len(face_1_results) == 0:
        raise Exception(f"图片[{image_path_1}]未识别到人脸")
    if len(face_2_results) == 0:
        raise Exception(f"图片[{image_path_2}]未识别到人脸")
    
    area = 0
    face_1 = None
    face_2 = None
    # 分别筛选面积最大的人脸进行人脸对比
    for face in face_1_results:
        a = (face.right() - face.left()) * (face.bottom() - face.top())
        if a > area:
            face_1 = face
            area = a
    area = 0
    for face in face_2_results:
        a = (face.right() - face.left()) * (face.bottom() - face.top())
        if a > area:
            face_2 = face
            area = a
            
    # 显示对比人脸
    cv2.rectangle(image_1, (face_1.left(), face_1.top()), (face_1.right(), face_1.bottom()),
                  (0, 0, 255), 1)
    cv2.imshow('image_1', image_1)
    cv2.rectangle(image_2, (face_2.left(), face_2.top()), (face_2.right(), face_2.bottom()),
                  (0, 0, 255), 1)
    cv2.imshow('image_2', image_2)
    cv2.waitKey(0)
    
    # 提取68个特征点
    shape = shape_predictor(image_1, face_1)
    # 计算人脸128维向量
    descriptor_1 = face_rec_model.compute_face_descriptor(image_1, shape)
    
    # 提取68个特征点
    shape = shape_predictor(image_2, face_2)
    # 计算人脸128维向量
    descriptor_2 = face_rec_model.compute_face_descriptor(image_2, shape)
    
    # 欧式距离 -- 三维示例：distance=sqrt((x1−x2)**2+(y1−y2)**2+(z1−z2)**2)
    sum = 0
    for i in range(len(descriptor_1)):
        sum += (descriptor_1[i] - descriptor_2[i]) ** 2
    import numpy as np
    distance = np.sqrt(sum)
    print(f"distance: {distance}")
    # 推荐0.6阈值 适用于成人
    if distance < 0.5:
        print('是同一个人')
    else:
        print('不是同一个人')
        
        
if __name__ == '__main__':
    image_path_1_1 = './image_test/76020190518171824_800008166198661.jpg'
    image_path_1_2 = './image_test/1558171552454_760.jpg'
    image_path_2 = './image_test/1551346226937_757_1557298319424.jpg'
    image_path_3 = './image_test/201705230909503371641_600_0.jpg'
    image_path_4 = './image_test/group.jpg'
    image_path_5_1 = './image_test/ym_1.jpg'
    image_path_5_2 = './image_test/ym_2.jpg'
    
    # face_detector(image_path_1_1)
    # face_detector(image_path_1_2)
    # shape_predictor(image_path_4)
    # face_detector_cnn(image_path_4)
    # single_target_tracker()
    face_contrast(image_path_5_1, image_path_5_2)