'''
* @author [daweihao]
* @version [2021-08-03]
* 〈亚洲人脸数据集按照男女分类〉
'''
import cv2
import os
import shutil

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # print(confidence)
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes


def genderDetect(img_path):
    '''
    性别检测
    :param img_path:
    :return:
    '''
    global gender
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"

    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    genderList = ['Male', 'Female']

    # Load network
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    # Open a video file or an image file or a camera stream
    padding = 20
    # Read frame
    frame = cv2.imread(img_path)
    frameFace, bboxes = getFaceBox(faceNet, frame)
    # print(bboxes)
    bbox = bboxes[0]
    # for bbox in bboxes:
        # print(bbox)
    face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
           max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
        # print("Gender Output : {}".format(genderPreds))
        # print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
        # cv2.imshow("Age Gender Demo", frameFace)

    return gender


if __name__ == '__main__':
    asian_dir = '/data/data_hao/style_gan2'
    man_dir = '/data/data_hao/style_gan2/man/'
    woman_dir = '/data/data_hao/style_gan2/woman/'
    for img_name in os.listdir(asian_dir):
        img_path = asian_dir + str('/') + img_name
        try:
            gender = genderDetect(img_path)

            if gender == 'Male':
                shutil.move(img_path, man_dir)
            elif gender == 'Female':
                shutil.move(img_path, woman_dir)
        except Exception:
            pass
        continue

    # # 亚洲人脸 男生
    # man_dir = '/data/data_hao/style_gan2/man/'
    # asian_dir = '/data/data_hao/style_gan2'
    # for img in os.listdir(man_dir):
    #     index = img.split('.jpg')[0].split('_')[0]
    #     for as_img in os.listdir(asian_dir):
    #         img_path = asian_dir + str('/') + as_img
    #         as_index = as_img.split('.jpg')[0].split('_')[0]
    #         if index == as_index:
    #             shutil.move(img_path, man_dir)
    #         else:
    #             pass
    #
    # # 亚洲人脸 女生
    # woman_dir = '/data/data_hao/style_gan2/woman/'
    # asian_dir = '/data/data_hao/style_gan2'
    # for img in os.listdir(woman_dir):
    #     index = img.split('.jpg')[0].split('_')[0]
    #     for as_img in os.listdir(asian_dir):
    #         img_path = asian_dir + str('/') + as_img
    #         as_index = as_img.split('.jpg')[0].split('_')[0]
    #         if index == as_index:
    #             shutil.move(img_path, woman_dir)
    #         else:
    #             pass
