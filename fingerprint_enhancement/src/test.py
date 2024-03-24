import cv2

# 读取图像
image = cv2.imread(r'D:\fingerprint_enhancement\img\finger6.png')

# 计算裁剪尺寸，即原图像宽高的80%
crop_width = int(image.shape[1] * 0.8)
crop_height = int(image.shape[0] * 0.8)

# 计算中心点坐标
center_x = int(image.shape[1] / 2)
center_y = int(image.shape[0] / 2)

# 计算裁剪矩形的左上角坐标
x = center_x - crop_width // 2
y = center_y - crop_height // 2

# 裁剪图像
cropped_image = image[y:y+crop_height, x:x+crop_width]

# 保存裁剪后的图像
cv2.imwrite('cropped_image.png', cropped_image)