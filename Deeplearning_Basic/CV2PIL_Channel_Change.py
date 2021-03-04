for i in range(img_channels):
    img_patch[:, :, i] = np.clip(img_patch[:, :, i], 0, 255)
