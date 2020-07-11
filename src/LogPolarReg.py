from scipy import misc, ndimage, signal
import numpy as np
import matplotlib.pyplot as plt


def log_polar_transform(img):
    im_center = np.array(img.shape) / 2
    log_r, theta = np.meshgrid(
        np.linspace(1, np.log(350), 250), np.linspace(-3.14159, 3.14159, 250),
    )
    xv = im_center[0] + np.exp(log_r) * np.sin(theta)
    yv = im_center[1] + np.exp(log_r) * np.cos(theta)
    log_polar_img = ndimage.map_coordinates(img, [xv, yv])
    return log_polar_img


img = misc.ascent()
img_sub = img[len(img) // 4 : 3 * len(img) // 4 : 2, len(img) // 4 : 3 * len(img) // 4 : 2]
big_img_sub = ndimage.zoom(img_sub, 0.5 + np.random.rand() * 10)
img_patch = ndimage.rotate(big_img_sub, np.random.rand() * 360, reshape=True)

fig = plt.figure(figsize=(10, 20))
[ax1, ax2], [ax3, ax4], [ax5, ax6] = fig.subplots(3, 2)

ax1.imshow(img, cmap="gray")
ax2.imshow(img_patch, cmap="gray")


log_polar_img = log_polar_transform(img)
log_polar_img_patch = log_polar_transform(img_patch)

ax3.imshow(log_polar_img, cmap="gray")
ax4.imshow(log_polar_img_patch, cmap="gray")

fig.set_tight_layout(True)

corr = signal.correlate(
    np.concatenate((log_polar_img, log_polar_img, log_polar_img)),
    log_polar_img_patch,
    mode="same",
    method="fft",
)

corr_shape = signal.correlate(
    np.concatenate(
        (np.ones(log_polar_img.shape), np.ones(log_polar_img.shape), np.ones(log_polar_img.shape))
    ),
    np.ones(log_polar_img.shape),
    mode="same",
    method="fft",
)

corr = corr / corr_shape
corr = corr[len(corr) // 3 : 2 * len(corr) // 3, :]
angle_idx, log_r_idx = np.unravel_index(np.argmax(corr), corr.shape)  # find the match


ax5.imshow(corr, cmap="gray")
ax5.set_title("Cross-correlation")


angle_vec = np.linspace(-3.14159, 3.14159, 250)
log_r_vec = np.linspace(1, np.log(350), 250)
angle = angle_vec[angle_idx] * 180 / 3.14159
r_scale = np.exp(log_r_vec[log_r_idx] - log_r_vec[len(log_r_vec) // 2])

reg_patch = ndimage.rotate(img_patch, -angle, reshape=True)
reg_patch = ndimage.zoom(reg_patch, r_scale)


ax6.imshow(reg_patch, cmap="gray")
ax6.set_title("adjusted")


plt.show()

