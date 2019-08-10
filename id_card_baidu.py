
import requests
import json
import base64
import cv2
import urllib

def get_access_token(ak, sk):
    """
    获取access_token
    :param ak: str API_KEY
    :param sk: str Secret Key
    :return: str access_token
    """
    url = 'https://aip.baidubce.com/oauth/2.0/token'
    data = {
        'grant_type': 'client_credentials',
        'client_id': ak,
        'client_secret': sk
    }
    headers = {
        'Content-Type': 'application/json; charset=UTF-8'
    }
    response = requests.post(url, data=data, headers=headers)
    if response.status_code == 200:
        return response.json()["access_token"]
    raise Exception(f"get_access_token error, 状态码: {response.status_code}")


def recognize_id_card(access_token, image, id_card_side, detect_direction=True, detect_risk=False):
    """
    身份证识别
    :param access_token:
    :param image: str 图片的base64格式
    :param id_card_side: str front：身份证含照片的一面；back：身份证带国徽的一面
    :param detect_direction: bool 是否检测图像旋转角度，默认检测，即：true
    :param detect_risk: bool 是否开启身份证风险类型(身份证复印件、临时身份证、身份证翻拍、修改过的身份证)功能，默认不开启，即：false
    :return:json
    """
    url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/idcard'
    data = {
        'access_token': access_token,
        'image': image,
        'id_card_side': id_card_side,
        'detect_direction': detect_direction,
        'detect_risk': detect_risk,
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = requests.post(url, data=data, headers=headers)
    if response.status_code == 200:
        if "error_code" in response.json():
            raise Exception(f"recognize_id_card error, error_code: {response.json()['error_code']},"
                            f" error_msg: {response.json()['error_msg']}")
        return response.json()
    raise Exception(f"recognize_id_card error, 状态码: {response.status_code}")


def is_has_face(access_token, image, image_type='BASE64', face_field="", max_face_num=1, face_type="LIVE", liveness_control="NONE"):
    """
    检测是否有人脸,用于判断是否是身份证桌面
    :param access_token: str
    :param image: str 图片的base64格式
    :param image_type: str 图片类型
        BASE64:图片的base64值，base64编码后的图片数据，编码后的图片大小不超过2M；
        URL:图片的 URL地址( 可能由于网络等原因导致下载图片时间过长)；
        FACE_TOKEN: 人脸图片的唯一标识，调用人脸检测接口时，会为每个人脸图片赋予一个唯一的FACE_TOKEN，同一张图片多次检测得到的FACE_TOKEN是同一个
    :param face_field: str 包括age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities信息，逗号分隔，默认只返回人脸框、概率和旋转角度
    :param max_face_num: int 最多处理人脸的数目，默认值为1，仅检测图片中面积最大的那个人脸
    :param face_type: str 人脸的类型
        LIVE表示生活照：通常为手机、相机拍摄的人像图片、或从网络获取的人像图片等
        IDCARD表示身份证芯片照：二代身份证内置芯片中的人像照片
        WATERMARK表示带水印证件照：一般为带水印的小图，如公安网小图
        CERT表示证件照片：如拍摄的身份证、工卡、护照、学生证等证件图片
        默认LIVE
    :param liveness_control: str 活体控制 检测结果中不符合要求的人脸会被过滤
        NONE: 不进行控制
        LOW:较低的活体要求(高通过率 低攻击拒绝率)
        NORMAL: 一般的活体要求(平衡的攻击拒绝率, 通过率)
        HIGH: 较高的活体要求(高攻击拒绝率 低通过率)
        默认NONE
    :return: bool
    """
    url = 'https://aip.baidubce.com/rest/2.0/face/v3/detect'
    data = {
        'access_token': access_token,
        'image': image,
        'image_type': image_type,
        'face_field': face_field,
        'max_face_num': max_face_num,
        'face_type': face_type,
        'liveness_control': liveness_control
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = requests.post(url, data=data, headers=headers)
    if response.status_code == 200:
        if response.json()['error_code'] != 0:
            raise Exception(f"is_has_face error, error_code: {response.json()['error_code']},"
                            f" error_msg: {response.json()['error_msg']}")
        # 没有检测到人脸
        if response.json()['result']['face_num'] == 0:
            return False
        return True
    raise Exception(f"is_has_face error, 状态码: {response.status_code}")
    
    
def face_contrast(access_token, image1_info, image2_info):
    """
    人脸对比
    :param access_token:
    :param image1_info: dict/json 第一张图片的信息
    :param image2_info: dict/json 第二张图片的信息
        示例：image1_info = {
            'image': str(image1, 'utf-8'), # 图片信息(总数据大小应小于10M)，图片上传方式根据image_type来判断
            'image_type': 'BASE64', # 图片类型 BASE64、URL、FACE_TOKEN
            'face_type': 'LIVE',    # 人脸的类型 LIVE（生活照）、IDCARD（身份证人像）、WATERMARK（带水印）、CERT（拍摄的证件照），默认LIVE
            'quality_control': 'NONE', # 图片质量控制 NONE: 不进行控制、LOW、NORMAL、HIGH 默认 NONE
            'liveness_control': 'NONE', # 活体检测控制 NONE: 不进行控制、LOW、NORMAL、HIGH 默认 NONE
        }
    :return: json
    """
    url = f'https://aip.baidubce.com/rest/2.0/face/v3/match?access_token={access_token}'
    data = [image1_info, image2_info]
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        # 人脸比对失败
        if json.loads(response.text)['error_code'] != 0:
            raise Exception(f"face_contrast error, error_code: {response.json()['error_code']},"
                            f" error_msg: {response.json()['error_msg']}")
        
        return response.json()
    raise Exception(f"is_has_face error, 状态码: {response.status_code}")
   
   
API_KEY = 'FVugBD3m8T820g6CYWPGRc3g'
SECRET_KEY = 'fQsbXfrKj60n0IymtbhKX4y3bwjQwo3L'

if __name__ == '__main__':
    # 获取access_token
    access_token = get_access_token(API_KEY, SECRET_KEY)
    # # 读取身份证图片
    # with open('./image_test/1558171552454_760.jpg', 'rb') as fp:
    #     image = base64.b64encode(fp.read())
    # fp.close()
    #
    # result = is_has_face(access_token, image, image_type='BASE64', face_type='CERT')
    # # if result['face_num'] > 0:
    # #     img = cv2.imread('./image_test/1551346226937_757_1557298319424.jpg', cv2.IMREAD_UNCHANGED)
    # #     for face in result['face_list']:
    # #         cv2.rectangle(img, (int(face['location']['left']), int(face['location']['top'])),
    # #                       (int(face['location']['left'] + face['location']['width']), int(face['location']['top'] + face['location']['height'])), 255)
    # #     cv2.imshow('人像选择', img)
    # #     cv2.waitKey(0)
    #
    # # 身份证带人脸面
    # if result:
    #     id_card_info = recognize_id_card(access_token=access_token, image=image, id_card_side='front')
    #     print(id_card_info)
    
    with open('./image_test/1_2.jpg', 'rb') as fp:
        image1 = base64.b64encode(fp.read())
    fp.close()
    
    with open('./image_test/1_1.jpg', 'rb') as fp:
        image2 = base64.b64encode(fp.read())
    fp.close()
    
    # with open('./test.jpg', 'rb') as fp:
    #     image_test = base64.b64encode(fp.read())
    # fp.close()
    # print(is_has_face(access_token, image_test, face_type='LIVE'))
    
    image1_info = {
        'image': str(image1, 'utf-8'),
        'image_type': 'BASE64',
        'face_type': 'LIVE',
        'quality_control': 'LOW',
        'liveness_control': 'NONE',
    }
    image2_info = {
        'image': str(image2, 'utf-8'),
        'image_type': 'BASE64',
        'face_type': 'LIVE',
        'quality_control': 'LOW',
        'liveness_control': 'NONE',
    }
    result = face_contrast(access_token, image1_info, image2_info)
    print(result)