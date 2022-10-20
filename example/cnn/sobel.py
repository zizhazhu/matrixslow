import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matrixslow as ms

pic = matplotlib.image.imread('./data/pics/mondrian.jpg') / 255
w, h = pic.shape

sobel_v = ms.core.Variable(dim=(3, 3), init=False, trainable=False)
sobel_v.set_value(np.mat([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))

sobel_h = ms.core.Variable(dim=(3, 3), init=False, trainable=False)
sobel_h.set_value(np.mat([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))

img = ms.core.Variable(dim=(w, h), init=False, trainable=False)
img.set_value(np.mat(pic))

sobel_v_output = ms.ops.Convolve(img, sobel_v)
sobel_h_output = ms.ops.Convolve(img, sobel_h)

square_output = ms.ops.Add(
    ms.ops.Multiply(sobel_v_output, sobel_v_output),
    ms.ops.Multiply(sobel_h_output, sobel_h_output),
)

square_output.forward()
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(221)
ax.axis("off")
ax.imshow(img.value, cmap="gray")

ax = fig.add_subplot(222)
ax.axis("off")
ax.imshow(square_output.value, cmap="gray")

ax = fig.add_subplot(223)
ax.axis("off")
ax.imshow(sobel_v_output.value, cmap="gray")

ax = fig.add_subplot(224)
ax.axis("off")
ax.imshow(sobel_h_output.value, cmap="gray")

plt.show()
