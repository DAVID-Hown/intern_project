# 卡通&&漫画人脸训练数据合成方法
  *为适配K歌线上渲染流程，实现变身动漫效果。构建生成张大嘴巴的卡通人脸数据集，以配合GauGAN模型进行训练。构建生成张大嘴巴的卡通人脸数据集(根据人脸属性定制亚洲人脸->卡通化人脸处理->将美术提供的嘴巴模板与生成的卡通嘴巴进行图像融合)、训练优化GauGAN模型。涉及模型(StyleGAN、Facial Landmark detection、GauGAN)、基本的图像处理技术(dlib、Gaussian Blur、swap face)、模型压缩。*
### 1、生成亚洲人脸
- [generators-with-stylegan2](https://github.com/a312863063/generators-with-stylegan2)：人脸属性编辑器，共有21种人脸属性，属性幅度(每次十个不同幅度)可调。生成结果demo见./video_src
- [gender_detect.py](https://learnopencv.com/age-gender-classification-using-opencv-deep-learning-c-python/)：将亚洲人脸按照性别进行分类，男/女
### 2、卡通化亚洲人脸
- face_cartoon_onnx.py：将男/女亚洲人脸数据集卡通化
### 3、数据集预处理
- data_processing.py：男/女数据集以及美术男/女嘴巴预处理
- hand_landmouth.py：一种嘴巴手工打标方法
- [mouth_open.py](https://github.com/peterjpxie/detect_mouth_open)：从[ffhq](https://github.com/NVlabs/ffhq-dataset)数据集中选出大嘴巴亚洲人脸
- generate_video.py：选择一张亚洲人脸，调整亚洲人脸的21个属性，20个幅度，fps=10，得到四个视频./ video_demo
- split_image.py：1024x512 crop 512x512
### 4、融合&&blendshape
- [face-of-art](https://github.com/papulke/face-of-art)：精准定位卡通人脸五官区域(dlib对于个别卡通人脸关键点定位失效)
- gaussian_blend.py：高斯色块融合方法(第一种融合方法)
- [mouth_swap.py](https://github.com/spmallick/learnopencv/blob/master/FaceSwap/faceSwap.py)：嘴巴置换方法(第二种融合方法——卡通化效果不佳)
### 5、对齐方法
- align_face.py：将亚洲人脸按照卡通人脸眼睛水平距离对齐，然后裁剪至相同大小
