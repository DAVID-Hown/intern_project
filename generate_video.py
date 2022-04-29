'''
* @author [daweihao]
* @version [2021-08-04]
* 〈生成视频图像数据处理〉
'''
import cv2
import numpy as np
import onnxruntime
import os
from PIL import Image


class FaceCartoonTest:
    MODEL_PATH = "cartoon_gan_0804.onnx"

    def __init__(self, model_path=None):
        if model_path is not None:
            self.cartoon = onnxruntime.InferenceSession(model_path)
        else:
            self.cartoon = onnxruntime.InferenceSession(self.MODEL_PATH)

        _, _1, self.inp_h, self.inp_w = self.cartoon._sess.inputs_meta[0].shape

    def infer(self, frame):
        ori_h, ori_w, _ = frame.shape

        inp_img = self.input_preprocess(frame)
        cartoon_face = self.cartoon.run(["310"], {"input.1": inp_img})[0]
        cartoon_face = np.squeeze(cartoon_face)
        cartoon_face = cartoon_face.transpose([1, 2, 0])
        cartoon_face = ((cartoon_face + 1.) / 2) * 255.0

        cartoon_face = cartoon_face.astype(np.uint8)
        cartoon_face = cartoon_face[:, :, ::-1]
        return cv2.resize(cartoon_face, (ori_w, ori_h))

    def input_preprocess(self, img):
        img = cv2.resize(img, (self.inp_w, self.inp_h))
        nor = img / 255.0
        inp = (nor - 0.5) / 0.5
        inp = inp[:, :, ::-1]
        inp = inp.transpose([2, 0, 1])
        inp = inp.astype(np.float32)
        return inp[np.newaxis, :, :, :]


if __name__ == '__main__':
    # 遍历含有21种属性的亚洲人脸
    img_list = []
    file_path = 'D:\\20210712\\Test_demo\\results\\results'
    for dir in os.listdir(file_path):
        sub_dir_path = os.path.join(file_path, dir)
        for sub_file in os.listdir(sub_dir_path):
            img_path = os.path.join(sub_dir_path, sub_file)
            img_list.append(img_path)
    # print(img_list[1])
    # print(len(img_list))

    # 亚洲人脸卡通化
    car_list = []
    name_list = []
    fc = FaceCartoonTest()
    for i in range(len(img_list)):
        img_dir = img_list[i]
        path, file = os.path.split(img_dir)
        img_name = file[:-4]
        img = cv2.imread(img_dir)
        cartoon_img = fc.infer(img)
        cartoon_img_dir = 'D:\\20210712\\Test_demo\\results\\cartoon\\{0}.jpg'.format(str(img_name))
        cv2.imwrite(cartoon_img_dir, cartoon_img)
        car_list.append(cartoon_img_dir)
        name_list.append(img_name)

    # 拼接所有图片
    for i in range(len(img_list)):
        asian_img_path = img_list[i]
        cartoon_img_path = car_list[i]
        save_path = 'D:\\20210712\\Test_demo\\results\\join\\{0}.jpg'.format(str(name_list[i]))
        img1, img2 = Image.open(asian_img_path), Image.open(cartoon_img_path)

        # 统一图片尺寸，可以自定义设置（宽，高）
        img1 = img1.resize((512, 512), Image.ANTIALIAS)
        img2 = img2.resize((512, 512), Image.ANTIALIAS)
        size1, size2 = img1.size, img2.size

        joint = Image.new('RGB', (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save(save_path)

    # 控制属性创建10FPS视频
    size = (1024, 512)
    # 创建视频对象
    object = ['0000', '0008', '0010', '0013', '0016']
    for i in range(5):
        videowrite = cv2.VideoWriter(object[i] + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 10, size)

        for filename in [r'D:\20210712\Test_demo\results\join\{0}.jpg'.format(object[i] + '_' + str(j).zfill(3)) for j
                         in range(420)]:
            videowrite.write(cv2.imread(filename))

        videowrite.release()
    print('end!')

