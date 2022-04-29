#! /usr/bin/env python
'''
* @author [daweihao]
* @version [2021-08-03]
* 〈将卡通人脸嘴巴与美术提供的五官中的嘴巴进行置换〉
'''
import numpy as np
import os
import cv2
import dlib


def readPoints(path):
    '''
    Read points from text file
    :param path: 嘴巴关键点txt文件路径
    :return: 获取嘴巴关键点坐标列表
    '''
    # Create an array of points.
    points = []

    # Read points
    with open(path) as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))

    return points



def applyAffineTransform(src, srcTri, dstTri, size):
    '''
    Apply affine transform calculated using srcTri and dstTri to src and output an image of size.
    :param src:
    :param srcTri:
    :param dstTri:
    :param size:
    :return:
    '''
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


# calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    '''
    根据美术嘴巴的convex hull的点得到
    The next step in alignment is to do a Delaunay triangulation of the points on the convex hull.
    This allows us to divide the face into smaller parts.
    :param rect: 待覆盖卡通人脸的嘴巴区域
    :param points: 待覆盖卡通人脸的嘴巴convex hull中的点
    :return: a Delaunay triangulation of the points on the convex hull
    '''
    # create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList()

    delaunayTri = []

    pt = []

    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            # Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
                        # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunayTri


def warpTriangle(img1, img2, t1, t2):
    '''
    Warps and alpha blends triangular regions from img1 and img2 to img
    对美术嘴巴进行仿射变换，覆盖卡通人脸中的嘴巴，生成融合后的卡通人脸
    :param img1: 美术提供的卡通人脸图像
    :param img2: 待覆盖的卡通人脸图像
    :param t1: 美术嘴巴的三角顶点
    :param t2: 卡通人脸嘴巴的三角顶点
    :return: 融合后的卡通人脸
    '''
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


def getMouthLandmarks(img_path):
    '''
    得到卡通人脸的嘴巴轮廓关键点txt文件
    :param img_path: 人脸图像的路径
    :return: 嘴巴关键点txt文件路径
    '''
    # image = face_recognition.load_image_file(img_path)
    # face_landmarks_list = face_recognition.face_landmarks(image)
    # pil_image = Image.fromarray(image)
    # d = ImageDraw.Draw(pil_image)
    #
    # top_lip = face_landmarks_list[0]['top_lip']
    # bottom_lip = face_landmarks_list[0]['bottom_lip']
    #
    # for tup in top_lip:
    #     bottom_lip.append(tup)

    # 嘴巴关键点检测
    global mouth_landmark
    img = cv2.imread(img_path)

    predictor_model = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
    predictor = dlib.shape_predictor(predictor_model)

    # cv2读取图像
    img = cv2.imread(img_path)

    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 人脸数rects（rectangles）
    rects = detector(img_gray, 0)

    (path, file) = os.path.split(img_path)
    txt_path = path + str('/') + file[:-4] + '.txt'
    # print(txt_path)
    file_handle = open(txt_path, 'a')

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])

        for idx, point in enumerate(landmarks[49:68], 48):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            print(idx + 1, pos)
            pos_info = str(point[0, 0]) + ' ' + str(point[0, 1]) + '\n'
            file_handle.write(pos_info)
            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(img, pos, 3, color=(0, 255, 0))

        mouth_landmark = landmarks[49:68]

    file_handle.close()

    # 将得到的关键点位置信息按行写进txt文件中，并将文件名以图像名字命名
    # 获取卡通人脸图片的filename
    # if not os.path.exists(txt_path):
    #     with open(txt_path, mode='w', encoding='utf-8') as ff:
    #         for line in mouth_landmark:
    #             for a in line:
    #                 a = str(a)
    #                 ff.write(a)
    #                 ff.write('\t')
    #             ff.write('\n')

    cv2.imwrite(file[:-4] + str('_') + "draw.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    return txt_path


def startSwap(img1_path, img2_path):
    '''
    生成置换后带有美术嘴巴的卡通人脸
    :param img1_path: 形变对象 .png
    :param img2_path: 待覆盖对象 .jpg
    :return:
    '''
    # Read images
    # 对美术嘴巴图片进行预处理
    ori_img1 = cv2.imread(img1_path)
    img1 = cv2.resize(ori_img1, (512, 512), interpolation=cv2.INTER_AREA)
    path, file = os.path.split(img1_path)
    # print(path) # /data/data_hao/PFLD-pytorch/woman_face
    # print(file) # woman05.png
    filename = file.split('.png')[0]
    img1_crop_path = path + str('/') + filename + '.png'
    cv2.imwrite(img1_crop_path, img1, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    img2 = cv2.imread(img2_path)
    img1Warped = np.copy(img2)

    # get txt_file of corresponding points
    txt_path1 = getMouthLandmarks(img1_crop_path)
    # 如果美术人脸得到关键点有误，人工打标注得到嘴巴区域的关键点保存为txt文件
    # txt_path2 = '/data/data_hao/PFLD-pytorch/woman_face/woman04.txt'
    txt_path2 = getMouthLandmarks(img2_path)

    # Read array of corresponding points
    points1 = readPoints(txt_path1)
    points2 = readPoints(txt_path2)

    # Find convex hull
    # 寻找嘴巴凸包(将边界点顺序连接成多边形)
    hull1 = []
    hull2 = []

    #  In Computer Vision and Math jargon, the boundary of a collection of points or shape is called a “hull”.
    #  A boundary that does not have any concavities is called a “Convex Hull”.
    #  The convex hull of a set of points can be calculated using OpenCV’s convexHull function.
    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
    print(len(hullIndex))
    for i in range(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])

    # Find delanauy traingulation for convex hull points
    # 三角剖分，将嘴巴区域进行三角剖分，划分成多个微小区域
    sizeImg2 = img2.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    # do a Delaunay triangulation of the points on the convex hull
    dt = calculateDelaunayTriangles(rect, hull2)

    if len(dt) == 0:
        quit()

    # Apply affine transformation to Delaunay triangles
    # The final steps of face alignment to to consider corresponding triangles between the source face
    # and the target face, and affine warp the source face triangle onto the target face.
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])
        # 仿射变化对齐
        warpTriangle(img1, img1Warped, t1, t2)

    # How to seamlessly combine the two images?
    # Clone seamlessly

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))

    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

    return output


if __name__ == '__main__':
    # img1_path表示形变卡通人脸
    # img2_path表示待覆盖卡通人脸
    img1_path = '/data/data_hao/PFLD-pytorch/woman_face/woman04.png'
    img2_path = '/data/data_hao/PFLD-pytorch/0000_007.jpg'
    output = startSwap(img1_path, img2_path)
    cv2.imwrite('swap_mouth_new.jpg', output, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    # cv2.imshow("Face Swapped", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Make sure OpenCV is version 3.0 or above
    # (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    #
    # if int(major_ver) < 3 :
    #     print >>sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher'
    #     sys.exit(1)


