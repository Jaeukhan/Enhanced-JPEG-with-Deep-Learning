import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.fftpack import dct, idct
import utills
from PIL import Image


class Two_encoding:
    def __init__(self, p1, p2, qfactor):
        self.p1 = p1
        self.p2 = p2
        self.qfactor = qfactor

    def encoding(self):
        h, w, c = self.p1.shape

        pil_image = Image.fromarray(self.p2)
        img3 = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        img3 = np.array(img3)

        img_ycrcb = cv2.cvtColor(self.p1, cv2.COLOR_BGR2YCrCb)
        img_ycrcb2 = cv2.cvtColor(img3, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = img_ycrcb[:, :, 0], img_ycrcb[:, :, 1], img_ycrcb[:, :, 2]
        y2, cr2, cb2 = img_ycrcb2[:, :, 0], img_ycrcb2[:, :, 1], img_ycrcb2[:, :, 2]

        y_z = np.zeros([h, w])
        y_z2 = np.zeros([h, w])
        for i in range(h):
            for j in range(w):
                y_z[i, j] = y[i, j]
                y_z2[i, j] = y2[i, j]
        subcb = self.chromasubsampling(cb, h, w)
        subcr = self.chromasubsampling(cr, h, w)

        subcb2 = self.chromasubsampling(cb2, h, w)
        subcr2 = self.chromasubsampling(cr2, h, w)

        y_blocks = self.transform_to_block(y_z)
        cb_blocks = self.transform_to_block(subcb)
        cr_blocks = self.transform_to_block(subcr)

        y_blocks2 = self.transform_to_block(y_z2)
        cb_blocks2 = self.transform_to_block(subcb2)
        cr_blocks2 = self.transform_to_block(subcr2)

        y_diff = y_blocks2 - y_blocks
        cb_diff = cb_blocks2 - cb_blocks
        cr_diff = cr_blocks2 - cr_blocks

        y_blocks = self.dct2d(y_blocks)
        cb_blocks = self.dct2d(cb_blocks)
        cr_blocks = self.dct2d(cr_blocks)
        print("DCT :")
        print(y_blocks[0])

        y_qnt = self.quantization(y_blocks, 'y', self.qfactor)  # (4096, 8,8)
        cb_qnt = self.quantization(cb_blocks, 'c', self.qfactor)
        cr_qnt = self.quantization(cr_blocks, 'c', self.qfactor
                                   )
        print('qnt')
        print(y_qnt[0])
        y_zig = self.encode_zigzag(y_qnt)  # (4096, 64)
        cb_zig = self.encode_zigzag(cb_qnt)
        cr_zig = self.encode_zigzag(cr_qnt)
        print(np.array(y_zig).shape)

        y_blocks2 = self.dct2d(y_diff)
        cb_blocks2 = self.dct2d(cb_diff)
        cr_blocks2 = self.dct2d(cr_diff)
        print("diff_DCT :")
        print(y_blocks[0])

        y_qnt2 = self.quantization(y_blocks2, 'y', self.qfactor)  # (4096, 8,8)
        cb_qnt2 = self.quantization(cb_blocks2, 'c', self.qfactor)
        cr_qnt2 = self.quantization(cr_blocks2, 'c', self.qfactor
                                   )
        print('diff_qnt')
        print(y_qnt2[0])
        y_zig2 = self.encode_zigzag(y_qnt2)  # (4096, 64)
        cb_zig2 = self.encode_zigzag(cb_qnt2)
        cr_zig2 = self.encode_zigzag(cr_qnt2)
        print(np.array(y_zig).shape)

        return [y_zig, cb_zig, cr_zig], [y_zig2, cb_zig2, cr_zig2]

    def qualityfactor(self, matrix, Q=50):
        scalefactor = 1
        if Q < 50:
            scalefactor = 5000 / Q
        elif Q >= 50:
            scalefactor = 200 - 2 * Q

        x = np.floor((matrix * scalefactor + 50) / 100)
        if scalefactor == 0:
            x += 1
        # print(x)

        return x

    def encode_zigzag(self, matrix):  # matrix shape: (4096,8,8)
        bitstream = []
        for i, block in enumerate(matrix):  # (8, 8)
            new_block = block.reshape([64])[utills.zigzag_order].tolist()
            zero_count = (new_block.count(0)) - 1
            bitstream.append(new_block[:-1 * zero_count])
        return bitstream

    def quantization(self, blocks, type, Q):
        Q_y = self.qualityfactor(utills.Q_y, Q)
        Q_c = self.qualityfactor(utills.Q_c, Q)
        quant_block = np.zeros_like(blocks)
        if (type == 'y'):
            for i in range(len(blocks)):
                quant_block[i] = np.divide(blocks[i], Q_y).round().astype(np.float64)
        elif (type == 'c'):
            for i in range(len(blocks)):
                quant_block[i] = np.divide(blocks[i], Q_c).round().astype(np.float64)
        return quant_block

    def transform_to_block(self, img, blocksize=8):  # block size에 따른 Q_table interpolation과 downsampling.
        img_w, img_h = img.shape
        blocks = []
        for i in range(0, img_w, blocksize):
            for j in range(0, img_h, blocksize):
                blocks.append(img[i:i + blocksize, j:j + blocksize])
        blocks = np.array(blocks)
        print('8x8 block')
        print(blocks[0])
        print('-128: ')
        blocks = blocks - 128
        print(blocks[0])
        return blocks

    def dct2d(self, imgblocks):
        dctblock = np.zeros_like(imgblocks)
        for i in range(len(imgblocks)):
            dctblock[i] = dct(dct(imgblocks[i], axis=0, norm='ortho'), axis=1, norm='ortho')
        return dctblock

    def chromasubsampling(self, chroma, h, w):
        sub = np.zeros([int(h / 2), int(w / 2)])
        for i in range(0, h, 2):
            for j in range(0, w, 2):
                # if i % 2 == 0 and j % 2 == 0:
                sub[int(i / 2), int(j / 2)] = chroma[i, j]

        return sub


class Two_decoding:
    def __init__(self, img1, img2, qfactor):
        self.y1 = img1[0]
        self.cb1 = img1[1]
        self.cr1 = img1[2]
        self.y2 = img2[0]
        self.cb2 = img2[1]
        self.cr2 = img2[2]
        self.Q = qfactor

    def decoding(self):
        y_ord = self.decode_zigzag(self.y1)
        cb_ord = self.decode_zigzag(self.cb1)
        cr_ord = self.decode_zigzag(self.cr1)

        y_ord2 = self.decode_zigzag(self.y2)
        cb_ord2 = self.decode_zigzag(self.cb2)
        cr_ord2 = self.decode_zigzag(self.cr2)

        y_deq = self.dequantization(y_ord, 'y', self.Q)
        cb_deq = self.dequantization(cb_ord, 'c', self.Q)
        cr_deq = self.dequantization(cr_ord, 'c', self.Q)
        print(cb_deq.shape)
        print("inverse quantization")
        print(y_deq[0])

        y_deq2 = self.dequantization(y_ord2, 'y', self.Q)
        cb_deq2 = self.dequantization(cb_ord2, 'c', self.Q)
        cr_deq2 = self.dequantization(cr_ord2, 'c', self.Q)

        y_i = self.idct_2d(y_deq)
        cb_i = self.idct_2d(cb_deq)
        cr_i = self.idct_2d(cr_deq)
        print("inverse DCT")
        print(y_i[0])

        y_i2 = self.idct_2d(y_deq2)
        cb_i2 = self.idct_2d(cb_deq2)
        cr_i2 = self.idct_2d(cr_deq2)
        print("inverse DCT")
        print(y_i2[0])

        print("+128")
        print((y_i[0] + 128))

        print("+128")
        print((y_i2[0] + 128))

        new_yi2 = y_i + y_i2
        newcb_i2 = cb_i + cb_i2
        newcr_i2 = cr_i + cr_i2

        y_i = self.reconstruct_from_blocks(y_i)
        cb_i = self.reconstruct_from_blocks(cb_i)
        cr_i = self.reconstruct_from_blocks(cr_i)

        cb_i = self.chroma_interpolation(cb_i)
        cr_i = self.chroma_interpolation(cr_i)

        h, w = y_i.shape
        img = np.zeros([h, w, 3])
        img[:, :, 0], img[:, :, 1], img[:, :, 2] = y_i, cr_i, cb_i
        img = img.astype(np.uint8)
        reconimg = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

        y_i2 = self.reconstruct_from_blocks(new_yi2)
        cb_i2 = self.reconstruct_from_blocks(newcb_i2)
        cr_i2 = self.reconstruct_from_blocks(newcr_i2)

        cb_i2 = self.chroma_interpolation(cb_i2)
        cr_i2 = self.chroma_interpolation(cr_i2)

        h, w = y_i.shape
        img2 = np.zeros([h, w, 3])
        img2[:, :, 0], img2[:, :, 1], img2[:, :, 2] = y_i2, cr_i2, cb_i2
        img2 = img2.astype(np.uint8)
        reconimg2 = cv2.cvtColor(img2, cv2.COLOR_YCrCb2BGR)

        pil_image = Image.fromarray(reconimg2)
        img3 = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        reconimg2 = np.array(img3)

        return reconimg, reconimg2

    def dequantization(self, blocks, type, Q):
        Q_y = self.qualityfactor(utills.Q_y, Q)
        Q_c = self.qualityfactor(utills.Q_c, Q)
        quant_block = np.zeros_like(blocks)
        if (type == 'y'):
            for i in range(len(blocks)):
                quant_block[i] = np.multiply(blocks[i], Q_y).round().astype(np.float64)
        elif (type == 'c'):
            for i in range(len(blocks)):
                quant_block[i] = np.multiply(blocks[i], Q_c).round().astype(np.float64)
        return quant_block

    def idct_2d(self, blocks):
        idctblock = np.zeros_like(blocks)
        for i in range(len(blocks)):
            idctblock[i] = idct(idct(blocks[i], axis=0, norm='ortho'), axis=1, norm='ortho')
        return idctblock

    def chroma_interpolation(self, matrix):
        src = matrix.astype(np.uint8)
        matrix = cv2.resize(src, (len(matrix[0]) * 2, len(matrix[1]) * 2), interpolation=cv2.INTER_CUBIC)
        return matrix

    def reconstruct_from_blocks(self, blocks):
        total_lines = []
        N_blocks = int(len(blocks) ** 0.5)
        # print("N", N_blocks)
        # print(len(blocks))
        for n in range(0, len(blocks) - N_blocks + 1, N_blocks):
            res = np.concatenate(blocks[n: n + N_blocks], axis=1)
            # print(res.shape)
            total_lines.append(res)
            # print(np.array(total_lines).shape)
        blocks = np.concatenate(total_lines) + 128
        return blocks

    def decode_zigzag(self, matrix):
        matrix = np.array(matrix)
        # print(matrix[0])
        zigord1 = np.zeros((matrix.shape[0], 64))
        zigord = np.zeros((matrix.shape[0], 64))
        for i, block in enumerate(matrix):
            for j in range(len(block)):
                zigord1[i][j] = block[j]
            zigord[i][utills.zigzag_order] = zigord1[i]
            # print(zigord[i])
            # print(zigord[i].reshape(8,8))
        zigord = zigord.reshape(matrix.shape[0], 8, 8)
        print("received& convert zigzag")
        print(zigord[0])
        return zigord
    def qualityfactor(self, matrix, Q=50):
        scalefactor = 1
        if Q < 50:
            scalefactor = 5000 / Q
        elif Q >= 50:
            scalefactor = 200 - 2 * Q

        x = np.floor((matrix * scalefactor + 50) / 100)
        if scalefactor == 0:
            x += 1
        # print(x)

        return x
