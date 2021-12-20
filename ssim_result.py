import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.fftpack import dct, idct
import utills
from two_transit import Two_encoding, Two_decoding
import natsort


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def ssim_write(img1, img2, ord):
    ss = ssim(img1, img2)
    with open(f'{ord}_ssim.txt', 'a') as pn:
        pn.write(str(ss)+'\n')
    return ss

def calculate(img1, img2, ord):
    ori = cv2.imread(img1)
    img = cv2.imread(img2)
    ss = ssim_write(ori, img, ord)
    return ss
if __name__ == '__main__':
    N = 8
    qfactor = 90  # 1~100 올라가수록 filesize up.
    direc = "final"
    direc = "result/"+direc
    label = "img_512"

    lidir = os.listdir(direc)
    lidir = natsort.natsorted(lidir)

    labeldir = os.listdir(label)
    labeldir = natsort.natsorted(labeldir)
    print(labeldir) #5개
    print(lidir) #45개
    img_list = []  # 30 60 90 120 150

    for i, file in enumerate(labeldir):
        print(direc + '/' + file)

        ss1 = calculate(label + '/' + file, direc + '/' + lidir[i], "5")
        ss2 = calculate(label + '/' + file, direc + '/' + lidir[i+5], "10")
        ss3 = calculate(label + '/' + file, direc + '/' + lidir[i+10], "15")
        ss4 = calculate(label + '/' + file, direc + '/' + lidir[i+15], "20")
        ss5 = calculate(label + '/' + file, direc + '/' + lidir[i+20], "25")
        ss6 = calculate(label + '/' + file, direc + '/' + lidir[i+25], "30")
        ss7 = calculate(label + '/' + file, direc + '/' + lidir[i+30], "50")
        ss8 = calculate(label + '/' + file, direc + '/' + lidir[i+35], "70")
        ss9 = calculate(label + '/' + file, direc + '/' + lidir[i+40], "90")
    # 90 img 복원
