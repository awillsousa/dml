#inspired by https://blancosilva.wordpress.com/teaching/mathematical-imaging/denoising-wavelet-thresholding/
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pywt
import numpy as np
import cv2

pic_path= '../data/pics/'
leo_path = pic_path + 'leo.png'
ven_path= pic_path + 'venus.png'

def show_pics(imgs, label):
    fig = plt.figure()
    fig.subplots_adjust(top=0.85)
    fig.suptitle(label, fontsize=18)
    i = 1
    for img in imgs:
        a = fig.add_subplot(2, 3, i)
        a.set_axis_off()
        plt.imshow(img, cmap = cm.Greys_r)
        if i % 3 == 1:
            a.set_title('Original')
        elif i % 3 == 2:
            a.set_title('Noisy')
        else:
            a.set_title('Reconstructed')
        a.set_axis_off()
        i += 1
    plt.show()


def regularize_image(img):
    # to grayscale
    regImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # convert to float
    regImg = np.float32(regImg)
    regImg /= 255;
    return regImg


noiseSigma = 1.5
mode = 'db8'
level = None

def w2dReg(img, mode=mode, level=level, noiseSigma = noiseSigma):
    # compute coefficients
    noisy_img = add_noise(img, noiseSigma = noiseSigma)
    rec_coeffs = denoise(noisy_img, mode, level, noiseSigma)
    # reconstruction
    rec_img = pywt.waverec2(rec_coeffs, mode);
    return noisy_img, rec_img, len(rec_coeffs) - 1

def denoise(noisy_img, mode, level, noiseSigma):
    coeffs = pywt.wavedec2(noisy_img, mode, level=level)
    # Thresholding the detail (i.e. high frequency) coefficiens
    threshold = noiseSigma * np.sqrt(2 * np.log2(noisy_img.size))
    rec_coeffs = coeffs
    rec_coeffs[1:] = (pywt.threshold(i, value=threshold, mode="soft") for i in rec_coeffs[1:])
    return rec_coeffs

def add_noise(img, noiseSigma = noiseSigma ):
    return  img + np.random.normal(0, noiseSigma, size=img.shape)

img1 = regularize_image(cv2.imread(leo_path))
img2 = regularize_image(cv2.imread(ven_path))

wavImg2 = w2dReg(img2, mode=mode, level=level)
wavImg1 = w2dReg(img1, mode=mode, level=level)


parstr = 'noiseSigma='+str(noiseSigma) + ' , mode=' + mode + ' , level=' +str(wavImg1[2])

show_pics([img1, wavImg1[0], wavImg1[1], img2, wavImg2[0], wavImg2[1]], label=parstr)







