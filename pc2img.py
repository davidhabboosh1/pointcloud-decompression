import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

pc = np.random.rand(16 ** 2, 3)

def pc2img(pc):
    pc *= 255
    pc = pc.reshape(16, 16, 3)
    return pc

img = pc2img(pc)
img = Image.fromarray(img, mode='RGB')

# visualize image
plt.imshow(img)
plt.show()