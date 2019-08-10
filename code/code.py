import cv2
import pytesseract

if __name__ == '__main__':
    for i in range(1, 9):
        img_gray = cv2.imread(f'./{i}.png', 0)
        _, binary_pic = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        s = pytesseract.image_to_string(binary_pic, lang='eng', config="-psm 6")
        print(s)
        # cv2.imshow('img', binary_pic)
        # cv2.waitKey(0)
    
    