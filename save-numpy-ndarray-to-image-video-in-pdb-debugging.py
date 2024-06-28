
import cv2
import numpy as np

# 要将形状为 [275, 128, 128, 3] 的 NumPy 数组保存为视频，可以使用 opencv-python 库。这个库提供了方便的工具来处理图像和视频。以下是一个示例函数 save_video，它将 NumPy 数组保存为视频文件：
def save_video(array, filename='output_video.mp4', fps=30):
    # 检查数组的形状
    if len(array.shape) != 4 or array.shape[3] != 3:
        raise ValueError('Input array must have shape [frames, height, width, 3]')
    
    frames, height, width, channels = array.shape

    # 定义视频编码器和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(frames):
        frame = array[i]

        # 确保帧数据是 uint8 类型
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        video_writer.write(frame)
    
    # 释放 VideoWriter 对象
    video_writer.release()

# 示例数据
video_array = np.random.rand(275, 128, 128, 3)

# 保存视频
save_video(video_array, 'random_video.mp4')


import numpy as np
from matplotlib import pyplot as plt

# can be used in pdb pdbpp debugging session, to save something into an image for debugging
def save_image(array, filename='debug_image.png'):
    # 检查数组的形状
    if len(array.shape) == 2:
        # 灰度图像
        plt.imshow(array, cmap='gray')
    elif len(array.shape) == 3 and array.shape[2] == 3:
        # RGB 图像
        plt.imshow(array)
    elif len(array.shape) == 3 and array.shape[2] == 1:
        # 单通道图像，转换为二维数组
        plt.imshow(array[:, :, 0], cmap='gray')
    else:
        raise ValueError('Unsupported image array shape: {}'.format(array.shape))
    plt.axis('off')  # 隐藏坐标轴
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# 示例图像数组
rgb_image_array = np.random.rand(128, 128, 3)
gray_image_array = np.random.rand(128, 128)

# 使用示例
save_image(rgb_image_array, 'rgb_image.png')
save_image(gray_image_array, 'gray_image.png')

