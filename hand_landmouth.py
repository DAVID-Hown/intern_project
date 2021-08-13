# -*- coding:utf-8 -*-
import cv2

# 读入美术图像
img = cv2.imread("D:\\20210712\\PFLD-pytorch\\woman04.png")
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # xy = "%d,%d" % (x, y)
        # 人工打标注
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        # 将嘴巴关键点写入txt文件中
        txt_path = "D:\\20210712\\PFLD-pytorch\\mouth04.txt"
        file_handle = open(txt_path, 'a')
        file_handle.write(str((x, y)[0]) + ' ' + str((x, y)[1]) + '\n')

        # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
        #             1.0, (0, 0, 0), thickness=1)
        print((x, y))
        cv2.imshow("image", img)
        file_handle.close()

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.moveWindow("image", 512, 512)
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

while (1):
    cv2.imshow("image", img)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()