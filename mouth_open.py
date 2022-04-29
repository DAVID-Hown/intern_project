'''
* @author [daweihao]
* @version [2021-08-03]
* <检测嘴巴是否张开>
'''
import math
from PIL import Image, ImageDraw
import face_recognition
import os
import shutil
import cv2


def get_lip_height(lip):
    sum = 0
    for i in [2, 3, 4]:
        # distance between two near points up and down
        distance = math.sqrt((lip[i][0] - lip[12 - i][0]) ** 2 +
                             (lip[i][1] - lip[12 - i][1]) ** 2)
        sum += distance
    return sum / 3


def get_mouth_height(top_lip, bottom_lip):
    sum = 0
    for i in [8, 9, 10]:
        # distance between two near points up and down
        distance = math.sqrt((top_lip[i][0] - bottom_lip[18 - i][0]) ** 2 +
                             (top_lip[i][1] - bottom_lip[18 - i][1]) ** 2)
        sum += distance
    return sum / 3


def check_mouth_open(top_lip, bottom_lip):
    top_lip_height = get_lip_height(top_lip)
    bottom_lip_height = get_lip_height(bottom_lip)
    mouth_height = get_mouth_height(top_lip, bottom_lip)

    # if mouth is open more than lip height * ratio, return true.
    ratio = 0.5
    if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
        return True
    else:
        return False


if __name__ == "__main__":

    # ffhq_path = '/data/data_hao/detect_mouth_open/images1024x1024'
    # for image_file in os.listdir(ffhq_path):
    #     try:
    #         image = face_recognition.load_image_file(ffhq_path + str('/') + image_file)
    #         face_landmarks_list = face_recognition.face_landmarks(image)
    #         pil_image = Image.fromarray(image)
    #         d = ImageDraw.Draw(pil_image)
    #
    #         top_lip = face_landmarks_list[0]['top_lip']
    #         bottom_lip = face_landmarks_list[0]['bottom_lip']
    #         print(top_lip)
    #         print(bottom_lip)
    #
    #
    #
    #         top_lip_height = get_lip_height(top_lip)
    #         bottom_lip_height = get_lip_height(bottom_lip)
    #         mouth_height = get_mouth_height(top_lip, bottom_lip)
    #
    #         # if mouth is open more than lip height * ratio, return true.
    #         ratio = 0.5
    #         if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
    #             # print("open")
    #             shutil.move(ffhq_path + str('/') + image_file, '/data/data_hao/detect_mouth_open/ffhq_open_mouth')
    #         else:
    #             pass
    #     except Exception:
    #         print('IndexError: list index out of range')
    #     continue
    # /data/data_hao/PFLD-pytorch/03.png
    # img = cv2.imread('/data/data_hao/PFLD-pytorch/woman_face/woman04.png')
    # img = cv2.resize(img, (512, 512))
    # cv2.imwrite('/data/data_hao/PFLD-pytorch/woman_crop04.jpg', img)
    # image = face_recognition.load_image_file('/data/data_hao/PFLD-pytorch/0000_006.jpg')
    image = face_recognition.load_image_file('/data/data_hao/PFLD-pytorch/woman_face/woman04.png')
    face_landmarks_list = face_recognition.face_landmarks(image)
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    print(face_landmarks_list)
    top_lip = face_landmarks_list[0]['top_lip']
    bottom_lip = face_landmarks_list[0]['bottom_lip']
    print(top_lip)
    print(bottom_lip)

    for tup in top_lip:
        bottom_lip.append(tup)
    # print(set(bottom_lip))  # 去重
    # landmark = list(set(bottom_lip))
    # print(landmark)
    # 将得到的关键点位置信息按行写进txt文件中，并将文件名以图像名字命名
    fw = open('04.txt', 'w')
    for line in list(set(bottom_lip)):
        for a in line:
            a = str(a)
            fw.write(a)
            fw.write('\t')
        fw.write('\n')
    fw.close()

    # fw.writelines(["%s\n" % item for item in list(set(bottom_lip))])
