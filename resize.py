#!/usr/bin/env python3
import os
from PIL import Image

SRC_DIR = '/home/HwHiAiUser/gp/DATASETS/test0909/imgs'
DST_DIR = '/home/HwHiAiUser/gp/DATASETS/test0909/imgs_640'
TARGET_SIZE = (640, 640)

def resize_keep_aspect(img: Image.Image) -> Image.Image:
    """等比例缩放+中心填充到目标尺寸"""
    img.thumbnail(TARGET_SIZE, Image.LANCZOS)          # 先缩放到“能放进去”的最大尺寸
    w, h = img.size
    # 计算填充边框
    pad_left = (TARGET_SIZE[0] - w) // 2
    pad_top  = (TARGET_SIZE[1] - h) // 2
    new_img = Image.new('RGB', TARGET_SIZE, (114, 114, 114))  # YOLO 常用灰边
    new_img.paste(img, (pad_left, pad_top))
    return new_img

def main():
    os.makedirs(DST_DIR, exist_ok=True)
    supported = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    for name in os.listdir(SRC_DIR):
        if name.lower().endswith(supported):
            src_path = os.path.join(SRC_DIR, name)
            dst_path = os.path.join(DST_DIR, name)
            with Image.open(src_path) as im:
                im = im.convert('RGB')          # 统一通道数
                new_im = resize_keep_aspect(im)
                new_im.save(dst_path, quality=95)
            print(f'Saved {dst_path}')

if __name__ == '__main__':
    main()