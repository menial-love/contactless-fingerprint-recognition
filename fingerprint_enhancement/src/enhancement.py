import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm
from utils import adjust_gamma, rgb2gray, image_enhance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str, required=True,
                        help="Path to origin image folder (absolute path)")

    args = parser.parse_args()
    origin_image_folder = args.i
    for img_name in tqdm(os.listdir(origin_image_folder)):
        img_path = os.path.join(origin_image_folder, img_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        crop_width = int(img.shape[1] * 0.85)
        crop_height = int(img.shape[0] * 0.85)
        # 计算中心点坐标
        center_x = int(img.shape[1] / 2)
        center_y = int(img.shape[0] / 2)
        # 计算裁剪矩形的左上角坐标
        x = center_x - crop_width // 2
        y = center_y - crop_height // 2
        # 裁剪图像
        cropped_image = img[y:y + crop_height, x:x + crop_width]
        # step 1 CLAHE apply
        img_gamma = adjust_gamma(cropped_image, gamma=2.2)
        img_gray = rgb2gray(img_gamma, np.asarray([0.2126, 0.7152, 0.0722]).astype(np.float32))
        cv2.imwrite('tmp.jpg', img_gray)
        img_gray = cv2.imread('tmp.jpg', 0)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(16, 16))
        cl1 = clahe.apply(img_gray)
        cv2.normalize(cl1, cl1, 0, 255, cv2.NORM_MINMAX)
        img_en = cv2.convertScaleAbs(cl1)
        os.remove('tmp.jpg')
        # step2 gabor filter
        enhanced_img = image_enhance(img_en) * 255
        # step3 gray-level inversion
        for i in range(len(enhanced_img)):
            for j in range(len(enhanced_img[0])):
                enhanced_img[i][j] = 255 - enhanced_img[i][j]
        cv2.imwrite('../out/' + 'enhanced_' + img_name, enhanced_img)
