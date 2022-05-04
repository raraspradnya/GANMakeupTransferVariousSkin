import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

format = [".jpg",".png",".jpeg"]

no_makeup_ar = []
for (path, dirnames, filenames) in os.walk('Hasil Eksperimen/Referensi/no_makeup'):
    no_makeup_ar.extend(os.path.join(path, name) for name in filenames if name.endswith(tuple(format)))

BeautyGAN = []
for (path, dirnames, filenames) in os.walk('Hasil Eksperimen/BeautyGAN/'):
    BeautyGAN.extend(os.path.join(path, name) for name in filenames if name.endswith(tuple(format)))

cpm = []
for (path, dirnames, filenames) in os.walk('Hasil Eksperimen/CPM/'):
    cpm.extend(os.path.join(path, name) for name in filenames if name.endswith(tuple(format)))

dmt = []
for (path, dirnames, filenames) in os.walk('Hasil Eksperimen/DMT/'):
    dmt.extend(os.path.join(path, name) for name in filenames if name.endswith(tuple(format)))

test_images = [sorted(BeautyGAN), sorted(dmt), sorted(cpm)]
ssim_result = []
mse_result = []
no_makeup_ar = sorted(no_makeup_ar)
    
for j in range(len(test_images)):
    ssim_resultz = []
    mse_resultz = []
    for k in range(len(test_images[j])):
        no_makeup = cv2.imread(no_makeup_ar[k//4])
        no_makeup = cv2.resize(no_makeup, (256, 256))
        no_makeup = cv2.cvtColor(no_makeup, cv2.COLOR_BGR2GRAY)
        transfer = cv2.imread(test_images[j][k])
        transfer = cv2.cvtColor(transfer, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("no makeup", no_makeup)
        # cv2.imshow("transfer", transfer)
        # cv2.waitKey(0)
        # print(no_makeup.shape)
        # print(transfer.shape)

        ssim_res = round(ssim(no_makeup, transfer, data_range=transfer.max() - transfer.min()), 3)
        mse_res = round(mean_squared_error(no_makeup, transfer), 3)
        ssim_resultz.append(ssim_res)
        mse_resultz.append(mse_res)
    ssim_result.append(ssim_resultz)
    mse_result.append(mse_resultz)
    print(ssim_resultz)
    print(mse_resultz)


# img = img_as_float(data.camera())
# rows, cols = img.shape

# noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
# rng = np.random.default_rng()
# noise[rng.random(size=noise.shape) > 0.5] *= -1

# img_noise = img + noise
# img_const = img + abs(noise)

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4),
#                          sharex=True, sharey=True)
# ax = axes.ravel()

# mse_none = mean_squared_error(img, img)
# ssim_none = ssim(img, img, data_range=img.max() - img.min())

# mse_noise = mean_squared_error(img, img_noise)
# ssim_noise = ssim(img, img_noise,
#                   data_range=img_noise.max() - img_noise.min())

# mse_const = mean_squared_error(img, img_const)
# ssim_const = ssim(img, img_const,
#                   data_range=img_const.max() - img_const.min())

# ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
# ax[0].set_xlabel(f'MSE: {mse_none:.2f}, SSIM: {ssim_none:.2f}')
# ax[0].set_title('Original image')

# ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
# ax[1].set_xlabel(f'MSE: {mse_noise:.2f}, SSIM: {ssim_noise:.2f}')
# ax[1].set_title('Image with noise')

# ax[2].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
# ax[2].set_xlabel(f'MSE: {mse_const:.2f}, SSIM: {ssim_const:.2f}')
# ax[2].set_title('Image plus constant')

# plt.tight_layout()
# plt.show()