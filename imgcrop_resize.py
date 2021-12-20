import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy.fftpack import dct, idct

if __name__ == '__main__':
    rsize = 512  # 1~100 올라가수록 filesize up.
    direc = "img"

    lidir = os.listdir(direc)
    psnr_li = []
    ssim_li = []
    for i, file in enumerate(lidir):
        print(direc + '/' + file)
        img = Image.open(direc + '/' + file)
        img_resize = img.resize((rsize, rsize), Image.LANCZOS)
        img_resize.save(f'img_512/{file[:-5]}.png')
