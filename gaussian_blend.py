'''
* @author [daweihao]
* @version [2021-08-03]
* <将美术嘴巴融合到卡通人脸中，卡通化人脸数据集用美术提供的嘴巴进行替换>
* <高斯融合>
'''

import cv2
import numpy as np
import os
import dlib
from PIL import Image


def get_gaussian_filter(w, h, c_x, c_y, variance):
    '''
    高斯滤波
    :param w:
    :param h:
    :param c_x:
    :param c_y:
    :param variance:
    :return:
    '''
    # h, w = 1280, 720
    heatmap = np.zeros((h, w))
    # variance = 60
    mul = 1.5
    # c_x, c_y = (360, 640)
    for x_p in range(0, w):
        for y_p in range(0, h):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)

            exponent = dist_sq / 2.0 / variance / variance
            new_val = np.exp(-exponent) * mul
            new_val = min(1, max(0, new_val))

            heatmap[y_p, x_p] = new_val
    return heatmap


class FaceLandmark:
    '''
    获取嘴巴关键点，生成色块颜色
    '''
    def __init__(self):
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        # faces_path = "amazing_cartoon/0001_004.jpg"
        # faces_path = "sample_imgs/0144_006.jpg"
        # 加载dlib自带的人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        # 加载模型
        self.predictor = dlib.shape_predictor(predictor_path)

    def __call__(self, img):
        # 参数1表示对图片进行上采样一次，有利于检测到更多的人脸
        dets = self.detector(img, 1)
        shape = self.predictor(img, dets[0])
        landmark = np.array([[p.x, p.y] for p in shape.parts()])
        # 48 -> 60 是嘴巴的关键点
        xmin = np.min(landmark[48:60, 0])
        xmax = np.max(landmark[48:60, 0])
        ymin = np.min(landmark[48:60, 1])
        ymax = np.max(landmark[48:60, 1])

        # 获取色块颜色，8 是下巴的点
        color_bgr = [0, 0, 0]
        for y in range(ymax + 5, landmark[8, 1]):
            color_bgr[0] += img[int(y), (xmax + xmin) // 2, 0]
            color_bgr[1] += img[int(y), (xmax + xmin) // 2, 1]
            color_bgr[2] += img[int(y), (xmax + xmin) // 2, 2]

        div = landmark[8, 1] - (ymax + 5)
        color_bgr = list(map(lambda _: _ // div, color_bgr))

        return [xmin, xmax, ymin, ymax], color_bgr


def blend_img(img_path, art_mouth):
    '''
    图像融合
    :param img_path: 卡通人脸图像
    :param art_mouth: 美术嘴巴图像
    :return:
    '''
    # img_path .jpg文件
    # art_mouth .png文件
    fl = FaceLandmark()
    img = cv2.imread(img_path)
    box, color = fl(img)
    h, w, c = img.shape

    c_x = (box[1] + box[0]) // 2
    c_y = (box[2] + box[3]) // 2

    var = min(box[1] - box[0], box[3] - box[2])
    mask = get_gaussian_filter(w, h, c_x, c_y, var)
    cv2.imwrite('mask.jpg', mask)
    # 色块结合高斯 mask 融合
    for ri in range(h):
        for ci in range(w):
            img_color = img[ri, ci, :]
            mask_val = mask[ri, ci]
            blend_color = np.array([
                mask_val * color[0] + (1 - mask_val) * img_color[0],
                mask_val * color[1] + (1 - mask_val) * img_color[1],
                mask_val * color[2] + (1 - mask_val) * img_color[2],
            ])
            img[ri, ci, :] = blend_color.astype(np.uint8)
    cv2.imwrite('mask_img.jpg', img)
    # 把嘴巴贴上去，因为嘴巴的 box 和美术的 box 在尺寸上不一致
    # 读取alpha通道
    mouth_img = cv2.imread(art_mouth, cv2.IMREAD_UNCHANGED)
    mh, mw, _ = mouth_img.shape
    # 取嘴巴矩阵框最大那条边为基准，对应美术 05 的 w，需要根据不同情况来设置
    # 但有一个最简单的方法是生成多个 slide
    # [max_slide * 0.7, max_slide * 1.0, max_slide * 1.3, max_slide * 1.5]
    max_slide = max(box[1] - box[0], box[3] - box[2])
    factor = max_slide / mw
    new_w, new_h = int(max_slide), int(mh * factor)
    mouth_img = cv2.resize(mouth_img, (new_w, new_h), cv2.INTER_CUBIC)
    # 计算出嘴巴矩阵的中心点，来确定美术嘴巴放置的位置
    xmin = max(0, int(c_x - new_w / 2))
    xmax = min(w, int(c_x + new_w / 2))
    ymin = max(0, int(c_y - new_h / 2))
    ymax = min(h, int(c_y + new_h / 2))

    for ri in range(ymin, ymax):
        for ci in range(xmin, xmax):
            img_color = img[ri, ci, :]
            mask_val = mouth_img[ri - ymin, ci - xmin, 3].astype(np.float32) / 255.0
            mouth_color = mouth_img[ri - ymin, ci - xmin, :3]
            blend_color = np.array([
                mask_val * mouth_color[0] + (1 - mask_val) * img_color[0],
                mask_val * mouth_color[1] + (1 - mask_val) * img_color[1],
                mask_val * mouth_color[2] + (1 - mask_val) * img_color[2],
            ])
            img[ri, ci, :] = blend_color.astype(np.uint8)

    return img


def select_image(img_dir):
    '''
    从数据集中选取图片，按照奇数偶数分成两类(分别对应美术提供的嘴巴五种类型)
    :param img_dir: 卡通人脸图像路径
    '''
    for image in os.listdir(img_dir):
        # 0019_004.jpg
        # 0493_000.jpg
        last_num = image.split('.jpg')[0].split('_')[1][-1]
        # 选取奇数编号的卡通人脸，偶数编号的卡通人脸移除
        if (int(last_num) % 2) == 0:
            delete_path = img_dir + str('/') + image
            os.remove(delete_path)
            # print(delete_path)
        else:
            continue


def start_fusion(img_dir, art_mouth_path, save_path):
    '''
    对数据集中的所有图片进行融合
    :param img_dir: 卡通人脸图像
    :param art_mouth_path: 美术提供的嘴巴图像
    :param save_path: 融合后的保存路径
    '''
    # 创建 mouth_level : mouth_path
    # 根据选取的卡通路径是属于奇数还是偶数，选择打卡
    level = {'1': '01.png', '3': '02.png', '5': '03.png', '7': '04.png', '9': '05.png'}
    # level = {'0': '01.png', '2': '02.png', '4': '03.png', '6': '04.png', '8': '05.png'}
    # 遍历男生/女生卡通人脸
    for image in os.listdir(img_dir):
        index = image.split('.jpg')[0].split('_')[1][-1]
        # 按照index找到对应的美术嘴巴
        img_list = os.listdir(art_mouth_path)
        img_list.sort(key=lambda x: int(x[:-4]))  # 文件名按数字排序
        # 得到待融合的两张图片路径
        mouth_path = art_mouth_path + str('/') + level[index]
        img_path = img_dir + str('/') + image
        # print(mouth_path)
        # print(img_path)
        # 融合
        fusion = blend_img(img_path, mouth_path)
        cv2.imwrite(save_path + str('/') + image.split('.jpg')[0] + '.png', fusion,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            # print(fusion)
            # print(save_path + str('/') + image.split('.jpg')[0] + '.png')
        # except Exception:
        #     delete_path = img_dir + str('/') + image
        #     os.remove(delete_path)
        # continue


def join(asian_dir, cartoon_dir, save_path):
    '''
    将亚洲人脸与对应的融合后的卡通人脸进行拼接
    :param asian_dir: 亚洲人脸路径
    :param cartoon_dir: 融合后的卡通人脸路径
    :param save_path: 拼接图像保存路径
    '''
    for cartoon_img in os.listdir(cartoon_dir):
        asian_img = asian_dir + str('/') + cartoon_img.split('.png')[0] + str('.jpg')
        img1, img2 = Image.open(asian_img), Image.open(cartoon_dir + str('/') + cartoon_img)
        # 统一图片尺寸，可以自定义设置（宽，高）
        img1 = img1.resize((512, 512), Image.ANTIALIAS)
        img2 = img2.resize((512, 512), Image.ANTIALIAS)
        size1, size2 = img1.size, img2.size

        joint = Image.new('RGB', (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save(save_path + str('/') + cartoon_img)


if __name__ == '__main__':
    img_path = '/data/data_hao/PFLD-pytorch/0007_005.jpg'
    art_mouth= '/data/data_hao/PFLD-pytorch/mouth03.png'
    img = blend_img(img_path, art_mouth)
    cv2.imwrite('/data/data_hao/PFLD-pytorch/fusion.jpg', img)

    # # 处理图片
    # Asian_woman_img_dir = '/data/data_hao/seeprettyface-face_editor/amazing_tuning/Asian_woman'
    # # woamn_img_dir = '/data/data_hao/PFLD-pytorch/cartoon_face_database/woman'
    # # # select_image(woamn_img_dir)
    # Asian_man_img_dir = '/data/data_hao/seeprettyface-face_editor/amazing_tuning/Asian_man'

    # select_image(man_img_dir)

    # 对奇数编号男生进行融合
    # art_man_path = "/data/data_hao/PFLD-pytorch/man_art"
    # save_man_path = '/data/data_hao/data_base/man_fusion'
    # man_img_dir = '/data/data_hao/data_base/cartoon_man/odd_man'
    # start_fusion(man_img_dir, art_man_path, save_man_path)

    # 对偶数编号男生进行融合 需要调整start_fusion代码level字典
    # man_img_dir = '/data/data_hao/seeprettyface-face_editor/tuning_cartoon/man_cartoon/even_man'
    # start_fusion(man_img_dir, art_man_path, save_man_path)

    # # 横向拼接后图像保存路径
    # join_man_path = '/data/data_hao/PFLD-pytorch/save_path/man_join'
    # # 横向拼接 男生
    # join(Asian_man_img_dir, save_man_path, join_man_path)

    # art_woman_path = "/data/data_hao/PFLD-pytorch/woman_art"
    # woman_img_dir = '/data/data_hao/data_base/cartoon_woman/odd_woman'
    # save_woman_path = '/data/data_hao/data_base/woman_fusion'
    # start_fusion(woman_img_dir, art_woman_path, save_woman_path)

    # woamn_img_dir = '/data/data_hao/seeprettyface-face_editor/tuning_cartoon/man_cartoon/even_woman'
    # start_fusion(woamn_img_dir, art_woman_path, save_woman_path)

    # # # 横向拼接后图像保存路径
    # join_woman_path = '/data/data_hao/PFLD-pytorch/save_path/woman_join'
    # # # 横向拼接 男生

    # join(Asian_man_img_dir, save_woman_path, join_woman_path)
    # Asian_woman_img_dir = '/data/data_hao/data_base/style_gan2'
    # save_woman_path = '/data/data_hao/data_base/woman_fusion'
    # join_woman_path = '/data/data_hao/data_base/woman_join'
    # join(Asian_woman_img_dir, save_woman_path, join_woman_path)

