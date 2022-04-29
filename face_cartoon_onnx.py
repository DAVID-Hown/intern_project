'''
* @author [daweihao]
* @version [2021-07-15]
* 〈卡通化亚洲人脸图像〉
'''
import cv2
import numpy as np
import onnxruntime
import os

class FaceCartoon:
    sess = None
    # MODEL_PATH = "models/FaceCartoon.pb"
    # INPUT_SIZE = 256

    MODEL_PATH = "noseg_224_20_0331.onnx"

    def __init__(self):
        self.sess = onnxruntime.InferenceSession(self.MODEL_PATH)

    def infer(self, img):
        ori_h, ori_w, _ = img.shape

        inp_img = self.input_preprocess(img)
        cartoon_face = self.sess.run(["output"], {"input": inp_img})[0]
        cartoon_face = np.squeeze(cartoon_face)
        cartoon_face = cartoon_face.transpose([1, 2, 0])
        cartoon_face = ((cartoon_face + 1.) / 2) * 255.0

        cartoon_face = cartoon_face.astype(np.uint8)
        cartoon_face = cartoon_face[:, :, ::-1]
        return cv2.resize(cartoon_face, (ori_w, ori_h))

    def input_preprocess(self, img, size=224):
        img = cv2.resize(img, (size, size))
        nor = img / 255.0
        inp = (nor - 0.5) / 0.5
        inp = inp[:, :, ::-1]
        inp = inp.transpose([2, 0, 1])
        inp = inp.astype(np.float32)
        return inp[np.newaxis, :, :, :]


if __name__ == '__main__':
    fc = FaceCartoon()
    asiac_img = '/data/data_hao/PFLD-pytorch/00140.png'
    mg = cv2.imread(asiac_img)
    r = fc.infer(mg)
    r = cv2.resize(r, (512, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite('/data/data_hao/PFLD-pytorch/CARTOON.png', r)

    # 亚洲人脸数据集路径 男
    # man_img = '/data/data_hao/style_gan2/man'
    # # 生成的卡通人脸存放路径 男
    # Cartoon_img = '/data/data_hao/data_base/cartoon_man'
    # for image in os.listdir(man_img):
    #     image_path = os.path.join(man_img, image)
    #     img = cv2.imread(image_path)
    #     r = fc.infer(img)
    #     cv2.imwrite(Cartoon_img + str('/') + image.split('.jpg')[0] + '.jpg', r)

    # # 亚洲人脸数据集路径 男
    # woman_img = '/data/data_hao/style_gan2/woman'
    # # 生成的卡通人脸存放路径 男
    # Cartoon_img = '/data/data_hao/data_base/cartoon_woman'
    # for image in os.listdir(woman_img):
    #     image_path = os.path.join(woman_img, image)
    #     img = cv2.imread(image_path)
    #     r = fc.infer(img)
    #     cv2.imwrite(Cartoon_img + str('/') + image.split('.jpg')[0] + '.jpg', r)



