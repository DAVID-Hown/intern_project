# -*- coding: utf-8 -*-
'''
* @author [daweihao]
* @version [2021-08-09]
* <image size(1024, 512)-->(512, 512)>
'''

import cv2
import os
from PIL import Image

def split_img(img_dir, save_dir):
    """分离图片
        Args:
            img_dir:
            save_dir:
    """

    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        size = img.shape
        new_height = size[0]
        new_width = size[1] // 2
        split_img = img[0:new_height, 0:new_width]
        split_img_path = os.path.join(save_dir, img_file)
        cv2.imwrite(split_img_path, split_img)


if __name__ == '__main__':
    # img_dir = r'D:\20210809\cartoon_noseg_0807\cartoon_noseg_0727\train'
    # save_dir = r'D:\20210809\split_image\train'
    # split_img(img_dir, save_dir)

    # img_dir = r'D:\20210809\cartoon_noseg_0807\cartoon_noseg_0727\test'
    # save_dir = r'D:\20210809\split_image\test'
    # split_img(img_dir, save_dir)
    #
    # img_dir = r'D:\20210809\cartoon_noseg_0807\cartoon_noseg_0727\val'
    # save_dir = r'D:\20210809\split_image\val'
    # split_img(img_dir, save_dir)
    img = cv2.imread(r'D:\20210809\split_image\train\MX2mzRjuQB.jpg')
    crop_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

    cv2.imwrite('crop_asian.png', crop_img)





