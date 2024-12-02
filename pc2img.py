import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import DracoPy
from sklearn.metrics import mean_squared_error

np.random.seed(0)

IMG_SIZE = 16

def compress(pc, compression_level=0, quantization=14):
    if quantization == 0:
        compressed = DracoPy.encode(pc, compression_level=compression_level)
    else:
        compressed = DracoPy.encode(pc, compression_level=compression_level, quantization_bits=quantization)
    return compressed

def decompress(compressed):
    decompressed = DracoPy.decode(compressed).points
    return decompressed

def pc2img(pc):
    pc_cpy = pc[np.argsort(pc[:, 2])]
    pc_cpy = pc_cpy.reshape(IMG_SIZE, IMG_SIZE, 3)
    return pc_cpy

# Generate a random point cloud
pc = np.random.rand(IMG_SIZE ** 2, 3)
img = pc2img(pc)

# Compress and decompress the point cloud
img_compressed = compress(pc, 10, 30)
img_decompressed = decompress(img_compressed)
img_decompressed = pc2img(img_decompressed)

print(mean_squared_error(img.flatten(), img_decompressed.flatten()))

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image
ax1.imshow(img)
ax1.set_title('Original Image')
ax1.axis('off')  # Hide axes

# Display the decompressed image
ax2.imshow(img_decompressed)
ax2.set_title('Decompressed Image')
ax2.axis('off')  # Hide axes

# Adjust layout and show the plot
plt.tight_layout()
plt.show()