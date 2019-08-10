
import face_recognition
import cv2

if __name__ == '__main__':
    
    image_path_1_1 = './image_test/76020190518171824_800008166198661.jpg'
    image_path_1_2 = './image_test/1558171552454_760.jpg'
    image_path_2 = './image_test/1551346226937_757_1557298319424.jpg'
    image_path_3 = './image_test/201705230909503371641_600_0.jpg'
    image_path_4 = './image_test/group.jpg'
    image_path_5_1 = './image_test/ym_1.jpg'
    image_path_5_2 = './image_test/ym_2.jpg'
    image_path_6_1 = './image_test/1_1.jpg'
    image_path_6_2 = './image_test/1_2.jpg'
    image_path_7_1 = './image_test/2_1.jpg'
    image_path_7_2 = './image_test/2_2.jpg'
    image_path_8_1 = './image_test/3_1.jpg'
    image_path_8_2 = './image_test/3_2.jpg'
    image_path_9_1 = './image_test/4_1.jpg'
    image_path_9_2 = './image_test/4_2.jpg'
    # image = face_recognition.load_image_file(image_path_1_1)
    # # face_locations = face_recognition.face_locations(image, model='cnn')
    # face_locations = face_recognition.face_locations(image)
    #
    # r, g, b = cv2.split(image)  # 分离三个颜色通道
    # image = cv2.merge([b, g, r])  # 融合三个颜色通道生成新图片
    #
    # for face_location in face_locations:
    #     top, right, bottom, left = face_location
    #     cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)
    #
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    image_1 = face_recognition.load_image_file(image_path_8_2)
    image_2 = face_recognition.load_image_file('test.jpg')
    
    image_1_encodings = face_recognition.face_encodings(image_1)
    image_2_encoding = face_recognition.face_encodings(image_2)[0]
    
    # 第一个参数是list，第二个参数是一个对比人脸，tolerance默认为0.6
    results = face_recognition.compare_faces(image_1_encodings, image_2_encoding, tolerance=0.6)
    print(results)
    
    
    