from __future__ import absolute_import
import h5py
import matplotlib.pyplot as plt
import numpy as np
from supres import DSen2_20, DSen2_60
import sys
sys.path.append('../')
from utils.imresize import imresize


DATA_PATH = '../data/'


def readh5(fname, im60=False, imGT=False):
    with h5py.File(DATA_PATH+fname, 'r') as f:
        d10 = f['im10'][()].transpose()
        d20 = f['im20'][()].transpose()
        if im60:
            d60 = f['im60'][()].transpose()
            if not imGT:
                return d10, d20, d60
        if imGT:
            dGT = f['imGT'][()].transpose()
            if im60:
                return d10, d20, d60, dGT
            else:
                return d10, d20, dGT
        return d10, d20


def RMSE(x1, x2):
    diff = x1.astype(np.float64)-x2.astype(np.float64)
    rms = np.sqrt(np.mean(np.power(diff, 2)))
    print('RMSE: {:.4f}'.format(rms))
    return rms


if __name__ == '__main__':

    # Siberia, same area of Fig. 8 in the paper
    print('Siberia')
    im10, im20, imGT = readh5('S2B_MSIL1C_20170725_T43WFQ.mat', imGT=True)
    SR20 = DSen2_20(im10, im20)
    # Evaluation against the ground truth on the 20m resolution bands (simulated)
    print('DSen2:')
    RMSE(SR20, imGT)
    print('Bicubic:')
    RMSE(imresize(im20, 2), imGT)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    cax = ax.imshow(SR20[:, :, 2])
    fig.colorbar(cax)
    ax.set_title('Super-resolved band B6')

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    cax = plt.imshow(np.abs(SR20[:, :, 4]-imGT[:, :, 4]), vmin=0, vmax=200)
    fig.colorbar(cax)
    ax.set_title('Absolute differences to the GT, band B11')
    plt.show(block=False)

    #

    # South Africa, same area of Fig. 9 in the paper
    print('S. Africa')
    im10, im20, im60, imGT = readh5('S2A_MSIL1C_20171028_T34HCH.mat', im60=True, imGT=True)
    SR60 = DSen2_60(im10, im20, im60)
    # Evaluation against the ground truth on the 60m resolution bands (simulated)
    print('DSen2:')
    RMSE(SR60, imGT)
    print('Bicubic:')
    RMSE(imresize(im60, 6), imGT)

    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    cax = plt.imshow(np.abs(SR60[:, :, 1]-imGT[:, :, 1]), vmin=0, vmax=200)
    fig.colorbar(cax)
    ax.set_title('Absolute differences to the GT, band B9')
    plt.show(block=False)

    #

    # New York, same area of Fig. 10 (bottom) in the paper
    # Here using the very deep variable (VDSen2)
    print('New York')
    im10, im20, imGT = readh5('S2B_MSIL1C_20170928_T18TWL.mat', im60=False, imGT=True)
    SR20 = DSen2_20(im10, im20, deep=False)
    # Evaluation against the ground truth on the 20m resolution bands (simulated)
    print('DSen2:')
    RMSE(SR20,imGT)
    print('Bicubic:')
    RMSE(imresize(im20, 2), imGT)

    #

    # Malmo, Sweden, same area of Fig. 10 (top) in the paper
    print('Malmo, no ground truth')
    im10, im20, im60 = readh5('S2A_MSIL1C_20170527_T33UUB.mat', im60=True, imGT=False)

    SR20 = DSen2_20(im10, im20)
    SR60 = DSen2_60(im10, im20, im60)

    # No ground truth available, no simulation. Comparison to the low-res input
    fig = plt.figure(4)
    ax1 = fig.add_subplot(121)
    plt.imshow(im60[:, :, 0], vmin=np.min(im60[:, :, 0]), vmax=np.max(im60[:, :, 0]))
    ax1.set_title('Band B1, input 60m')
    ax2 = fig.add_subplot(122)
    plt.imshow(SR60[:, :, 0], vmin=np.min(im60[:, :, 0]), vmax=np.max(im60[:, :, 0]))
    ax2.set_title('Band B1, 10m super-resolution')
    plt.show(block=False)

    fig = plt.figure(5)
    ax1 = fig.add_subplot(121)
    plt.imshow(im20[:, :, 1], vmin=np.min(im20[:, :, 1]), vmax=np.max(im20[:, :, 1]))
    ax1.set_title('Band B6, input 20m')
    ax2 = fig.add_subplot(122)
    plt.imshow(SR20[:, :, 1], vmin=np.min(im20[:, :, 1]), vmax=np.max(im20[:, :, 1]))
    ax2.set_title('Band B6, 10m super-resolution')
    plt.show(block=False)

    #

    # Shark bay, Australia, same area of Fig. 10 (middle) in the paper
    print('Shark Bay, no ground truth')
    im10, im20, im60 = readh5('S2B_MSIL1C_20171022_T49JGM.mat', im60=True, imGT=False)
    SR20 = DSen2_20(im10, im20)
    SR60 = DSen2_60(im10, im20, im60)

    # Stretching the image for better visualization
    for i in range(SR60.shape[2]):
        a, b = np.percentile(SR60[:, :, i], (1, 99))
        SR60[:, :, i] = np.clip(SR60[:, :, i], a, b)
        SR60[:, :, i] = (SR60[:, :, i] - a) / (b - a)
        im60[:, :, i] = np.clip(im60[:, :, i], a, b)
        im60[:, :, i] = (im60[:, :, i] - a) / (b - a)
    imSR = SR60[:, :, [0, 1, 0]]
    im60s = im60[:, :, [0, 1, 0]]

    fig = plt.figure(6)
    ax1 = fig.add_subplot(121)
    plt.imshow(im60s)
    ax1.set_title('Color composite (B1,B9,B1) \n 60m input')
    ax2 = fig.add_subplot(122)
    plt.imshow(imSR)
    ax2.set_title('Color composite (B1,B9,B1) \n 10m super-resolution')

    # Stretching the image for better visualization
    imSR = SR20[:, :, [5, 3, 0]]
    im20s = im20[:, :, [5, 3, 0]]
    for i in range(imSR.shape[2]):
        a, b = np.percentile(imSR[:, :, i], (1, 99))
        imSR[:, :, i] = np.clip(imSR[:, :, i], a, b)
        imSR[:, :, i] = (imSR[:, :, i] - a) / (b - a)
        im20s[:, :, i] = np.clip(im20s[:, :, i], a, b)
        im20s[:, :, i] = (im20s[:, :, i] - a) / (b - a)

    fig = plt.figure(7)
    ax1 = fig.add_subplot(121)
    plt.imshow(im20s)
    ax1.set_title('Color composite (B12,B8a,B5) \n 20m input')
    ax2 = fig.add_subplot(122)
    plt.imshow(imSR)
    ax2.set_title('Color composite (B12,B8a,B5) \n 10m super-resolution')

    plt.show()
