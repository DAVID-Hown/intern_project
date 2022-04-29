'''
* @author [daweihao]
* @version [2021-07-26]
* <融合图像过程中的数据预处理>
'''
import cv2
import numpy as np
import os
import dlib
from PIL import Image
import shutil
import glob
from blend_img_david import blend_img

def start_fusion(img_dir, art_mouth_path, save_path):
    '''
    对数据集中的所有图片进行融合
    :param img_dir:
    :return:
    '''
    # 创建 mouth_level : mouth_path
    # level = {'1': '01.png', '3': '02.png', '5': '03.png', '7': '04.png', '9': '05.png'}
    level = {'0': '01.png', '2': '02.png', '4': '03.png', '6': '04.png', '8': '05.png'}
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
        try:
            fusion = blend_img(img_path, mouth_path)
            cv2.imwrite(save_path + str('/') + image.split('.jpg')[0] + '.png', fusion,
                        [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        except Exception:
            delete_path = img_dir + str('/') + image
            os.remove(delete_path)
        continue



def join(asian_dir, cartoon_dir, save_path):
    """
    :param png1: path
    :param png2: path
    :param flag: horizontal or vertical
    :return:
    """
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

if __name__ == "__main__":
    # cartoon_img_dir = '/data/data_hao/seeprettyface-face_editor/tuning_cartoon'
    # man_cartoon = '/data/data_hao/seeprettyface-face_editor/tuning_cartoon/man_cartoon'
    # woman_cartoon = '/data/data_hao/seeprettyface-face_editor/tuning_cartoon/woman_cartoon'
    # with open("man_number.txt", "r") as f:
    #     for line in f.readlines():
    #         line = line.strip('\n')  # 去掉列表中每一个元素的换行符
    #         # 0003_000
    #         for i in range(10):
    #             img_path = cartoon_img_dir + str('/') + line.split('_')[0] + str('_') + str(i).zfill(3) + '.jpg'
    #             shutil.move(img_path, man_cartoon)
    #             os.remove(img_path)


    # woman_cartoon_path = glob.glob(os.path.join(cartoon_img_dir + str('/'), '*.jpg'))
    # # print(woman_cartoon_path)
    # for i in range(len(woman_cartoon_path)):
    #     shutil.move(woman_cartoon_path[i], woman_cartoon)
    #     # os.remove(woman_cartoon_path[i])

    # man_img_dir = '/data/data_hao/data_base/cartoon_man'
    # for image in os.listdir(man_img_dir):
    #     # 0019_004.jpg
    #     # 0493_000.jpg
    #     try:
    #         last_num = image.split('.jpg')[0].split('_')[1][-1]
    #         # 选取偶数编号的卡通人脸
    #         if (int(last_num) % 2) == 0:
    #             even_path = man_img_dir + str('/') + image
    #             shutil.move(even_path, '/data/data_hao/data_base/cartoon_man/even_man')
    #             # print(delete_path)
    #     except Exception:
    #         print('wrong')
    #     continue
    #
    # man_img_dir = '/data/data_hao/data_base/cartoon_man'
    # for image in os.listdir(man_img_dir):
    #     # 0019_004.jpg
    #     # 0493_000.jpg
    #     try:
    #         last_num = image.split('.jpg')[0].split('_')[1][-1]
    #         # 选取奇数编号的卡通人脸
    #         if (int(last_num) % 2) != 0:
    #             odd_path = man_img_dir + str('/') + image
    #             shutil.move(odd_path, '/data/data_hao/data_base/cartoon_man/odd_man')
    #     except Exception:
    #         print('wrong')
    #     continue
    #
    # woman_img_dir = '/data/data_hao/data_base/cartoon_woman'
    # for image in os.listdir(woman_img_dir):
    #     # 0019_004.jpg
    #     # 0493_000.jpg
    #     try:
    #         last_num = image.split('.jpg')[0].split('_')[1][-1]
    #         # 选取偶数编号的卡通人脸
    #         if (int(last_num) % 2) == 0:
    #             even_path = woman_img_dir + str('/') + image
    #             shutil.move(even_path, '/data/data_hao/data_base/cartoon_woman/even_woman')
    #             # print(delete_path)
    #     except Exception:
    #         print('wrong')
    #     continue
    #
    # for image in os.listdir(woman_img_dir):
    #     # 0019_004.jpg
    #     # 0493_000.jpg
    #     try:
    #         last_num = image.split('.jpg')[0].split('_')[1][-1]
    #         # 选取奇数编号的卡通人脸
    #         if (int(last_num) % 2) != 0:
    #             odd_path = woman_img_dir + str('/') + image
    #             print(odd_path)
    #             shutil.move(odd_path, '/data/data_hao/data_base/cartoon_woman/odd_woman')
    #     except Exception:
    #         print('wrong')
    #     continue

    # art_man_path = "/data/data_hao/PFLD-pytorch/man_art"
    # save_man_path = '/data/data_hao/data_base/man_fusion'
    # # 男 偶
    # man_img_dir = '/data/data_hao/data_base/cartoon_man/even_man'
    # start_fusion(man_img_dir, art_man_path, save_man_path)

    # art_woman_path = "/data/data_hao/PFLD-pytorch/woman_art"
    # woman_img_dir = '/data/data_hao/data_base/cartoon_woman/even_woman'
    # save_woman_path = '/data/data_hao/data_base/woman_fusion'
    # start_fusion(woman_img_dir, art_woman_path, save_woman_path)


    # 横向拼接后图像保存路径
    Asian_man_img_dir = '/data/data_hao/data_base/style_gan2'
    save_man_path = '/data/data_hao/data_base/man_fusion'
    join_man_path = '/data/data_hao/data_base/man_join'
    join(Asian_man_img_dir, save_man_path, join_man_path)
