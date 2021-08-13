# -*- coding: utf-8 -*-
'''
* @author [v_daweihao]
* @version [2021-08-10]
* 〈按照眼睛关键点对齐人脸〉
* Reference:
    https://blog.csdn.net/weixin_35732969/article/details/83714492
    https://blog.csdn.net/u013841196/article/details/85720897
    https://zhuanlan.zhihu.com/p/32713815
    https://blog.csdn.net/qq_20622615/article/details/80929746
    https://blog.csdn.net/qq_36560894/article/details/105416273
    https://blog.csdn.net/weixin_35732969/article/details/83714492
'''

import cv2
import numpy as np
import os
import dlib
from PIL import Image
import shutil

def get_face_mark(img_path):
    '''
    代码功能：
    1. 用dlib人脸检测器检测出人脸，返回的人脸矩形框
    2. 对检测出的人脸进行关键点检测并用圈进行标记
    3. 将检测出的人脸关键点信息写到txt文本中
    :param img_path:
    :return:
    '''
    global landmarks
    predictor_model = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
    predictor = dlib.shape_predictor(predictor_model)

    # cv2读取图像
    test_img_path = img_path
    # output_pos_info = "test_img/Messi.txt"
    img = cv2.imread(test_img_path)
    # file_handle = open(output_pos_info, 'a')
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 人脸数rects
    rects = detector(img_gray, 0)
    print('人脸数：' + str(len(rects)))

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            # print(idx + 1, pos)
            # pos_info = str(point[0, 0]) + ' ' + str(point[0, 1]) + '\n'
            # file_handle.write(pos_info)
            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(img, pos, 3, color=(0, 255, 0))
            # 利用cv2.putText输出1-68
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, str(idx+1), pos, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    # file_handle.close()
    cv2.imwrite("face_keypoints.png", img)
    face_landmarks = []
    landmarks = [j for j in np.array(landmarks)]
    for i in range(68):
        face_landmarks.append((landmarks[i][0], landmarks[i][1]))

    return face_landmarks


def get_horizontal_offset(landmarks):
    '''获得眼睛水平方向offset
    :param landmarks: 68点人脸关键点
    :return:
    '''
    # get list landmarks of left and right eye
    left_eye = landmarks[37:43]
    right_eye = landmarks[43:49]

    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")

    # compute the horizontal distance between the eye centroids
    dx = right_eye_center[0] - left_eye_center[0]

    # # 计算左眼最左边关键点、右眼最右边关键点
    # left_eye_min = np.min(left_eye, axis=0).astype("int")[1]
    # right_eye_max = np.max(right_eye, axis=0).astype("int")[1]

    # 计算两只眼睛的水平距离
    # dx = right_eye_max[0] - left_eye_min[0]

    # calculate the center of 2 eyes
    # eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
    #               (left_eye_center[1] + right_eye_center[1]) // 2)

    return dx


def get_scaled_img(scale_ratio, img):
    '''对图片进行等比例缩放
    :param scale_ratio: dx_cartoon / dx_asian
    :param img:
    :return:
    '''
    # shape (高, 宽)
    h = img.shape[0]  # 图像的高
    w = img.shape[1]  # 图像的宽

    ori_ratio = w / h
    new_height = int(h * scale_ratio)
    new_width = int(new_height * ori_ratio)

    # (宽, 高)
    img_new = cv2.resize(img, (new_width, new_height))

    return img_new


def start_crop(split_img_dir, pcg_dir, scaled_dir, crop_dir, result_path, save_path):
    '''剪裁亚洲人脸
        按照卡通人脸眼睛水平距离缩放图片 --> scaled image
        对scaled image进行padding --> result --> resize (512, 512)
        join(asian image, cartoon image)
    :param split_img_dir:
    :param pcg_dir:
    :param scaled_dir:
    :param crop_dir:
    :param result_path:
    :param save_path:
    :return:
    '''

    for img in os.listdir(split_img_dir):
        try:
            img_path = os.path.join(split_img_dir, img)
            asian_img = cv2.imread(img_path)
            asian_face_landmarks = get_face_mark(img_path)
            dx_asian = get_horizontal_offset(asian_face_landmarks)

            cartoon_path = os.path.join(pcg_dir, img)
            cartoon_face_landmarks = get_face_mark(cartoon_path)
            dx_cartoon = get_horizontal_offset(cartoon_face_landmarks)

            cartoon_chin_landmarks = cartoon_face_landmarks[0:27]
            cartoon_chin_min = np.min(cartoon_chin_landmarks, axis=0).astype("int")[1]
            cartoon_chin_max = np.max(cartoon_chin_landmarks, axis=0).astype("int")[1]
            upper_offset = cartoon_chin_min
            bottom_offset = 512 - cartoon_chin_max

            # get scale ratio
            scale_ratio = dx_cartoon / dx_asian
            scale_img = get_scaled_img(scale_ratio, asian_img)
            scale_path = os.path.join(scaled_dir, img)
            cv2.imwrite(scale_path, scale_img)

            scaled_face_landmarks = get_face_mark(scale_path)
            chin_landmarks = scaled_face_landmarks[0:27]

            # 计算脸的轮廓 最上边/最下边点
            chin_min = np.min(chin_landmarks, axis=0).astype("int")[1]
            chin_max = np.max(chin_landmarks, axis=0).astype("int")[1]

            delta_upper = chin_min - upper_offset
            delta_bottom = chin_max + bottom_offset
            # 裁剪图片
            crop_img = scale_img[delta_upper:delta_bottom, :]
            crop_path = os.path.join(crop_dir, img)
            cv2.imwrite(crop_path, crop_img)

            # 先固定长边，左右两边补齐至正方形
            h, w = crop_img.shape[0], crop_img.shape[1]
            max_edge = max(h, w)
            min_edge = min(h, w)
            supple_length = max_edge - min_edge
            # 左右
            if h > w:
                left = supple_length // 2
                right = supple_length // 2
                pad_image = cv2.copyMakeBorder(crop_img, 0, 0, left, right, cv2.BORDER_REPLICATE, value=(0, 0, 0))
            else:
                top = supple_length // 2
                bottom = supple_length // 2
                pad_image = cv2.copyMakeBorder(crop_img, top, bottom, 0, 0, cv2.BORDER_REPLICATE, value=(0, 0, 0))

            result = cv2.resize(pad_image, (512, 512))
            padding_path = os.path.join(result_path, img)
            cv2.imwrite(padding_path, result)

            img1, img2 = Image.open(padding_path), Image.open(cartoon_path)
            size1, size2 = img1.size, img2.size
            joint = Image.new('RGB', (size1[0] + size2[0], size1[1]))
            loc1, loc2 = (0, 0), (size1[0], 0)
            joint.paste(img1, loc1)
            joint.paste(img2, loc2)
            joint.save(save_path + str('/') + img)
        except Exception:
            pass
        continue


def remove_wrong(pcg_dir, save_path, wrong_path):
    '''筛掉错误的数据
        去除关键点错误的图片
    :param pcg_dir:
    :param save_path:
    :param wrong_path:
    :return:
    '''

    for img in os.listdir(pcg_dir):
        cartoon_path = os.path.join(pcg_dir, img)
        cartoon_img = cv2.imread(cartoon_path)

        # 检测关键点是否存在
        predictor_model = 'shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
        predictor = dlib.shape_predictor(predictor_model)

        # 取灰度
        img_gray = cv2.cvtColor(cartoon_img, cv2.COLOR_RGB2GRAY)
        # 人脸数rects
        rects = detector(img_gray, 0)

        src_path = os.path.join(save_path, img)
        dst_path = os.path.join(wrong_path, img)

        # if not os.path.exists(src_path):
        #     continue
        # else:
        #     if len(rects) == 0:
        #         shutil.move(src_path, dst_path)
        if len(rects) == 0:
            shutil.move(src_path, dst_path)
        else:
            pass


if __name__ == '__main__':

    split_img_dir = '/data/data_hao/PFLD-pytorch/20200809/split_image/val'
    pcg_dir = '/data/data_hao/PFLD-pytorch/20200809/cartoon_pcg/cartoon_pcg_noseg_0807/val'
    scaled_dir = '/data/data_hao/PFLD-pytorch/20200809/scaled_img'
    crop_dir = '/data/data_hao/PFLD-pytorch/20200809/crop_img'
    result_path = '/data/data_hao/PFLD-pytorch/20200809/result'
    save_path = '/data/data_hao/PFLD-pytorch/20200809/join/val'
    wrong_path = '/data/data_hao/PFLD-pytorch/20200809/wrong_img'

    start_crop(split_img_dir, pcg_dir, scaled_dir, crop_dir, result_path, save_path)

    remove_wrong(pcg_dir, save_path, wrong_path)









