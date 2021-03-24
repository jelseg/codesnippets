
""" 
loading datasets
"""
from tensorflow.keras.preprocessing import image_dataset_from_directory

ds = image_dataset_from_directory(
    '../input/car-or-truck/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=1,
    shuffle=True,
)




"""
basic base:
1. filter = convultion
2.detect = relu activation
3. condense = maximum pooling
"""
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3,activation='relu'),
    #filters: the number of applied filters, kernel_size: the dimension of the kernel used in each filter (her 3x3)
    layers.MaxPool2D(pool_size=2)
    #-> less pixels -> less black pixels that don't contribute anything + intensify features, also destroys some positional info (can be good or bad)
])

#

"""
read and resize and show image
"""
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[400, 400])

img = tf.squeeze(image).numpy()
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# Reformat for batch compatibility.
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)



"""
if you want to do convolution and activation yourself:
    (just ass example to show the backend functions in tensorflow)
"""
import tensorflow as tf

kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])

plt.figure(figsize=(3, 3))
show_kernel(kernel)

#kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
#kernel = tf.cast(kernel, dtype=tf.float32)

image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    # we'll talk about these two in lesson 4!
    strides=1,
    padding='SAME',
)

plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.show()

mage_detect = tf.nn.relu(image_filter)

plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.show()

image_condense = tf.nn.pool(
    input=image_detect, # image in the Detect step above
    window_shape=(2, 2),
    pooling_type='MAX',
    strides=(2, 2), #every step skips one pixel
    padding='SAME', #how to handle borders, same-> adds 0s so result has same size, valid -> image will shrink
)

plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.show()



"""
global average pool:
replace flatten and some or all intermediate dense layers 
can drasticly lower number of parameters
all feature maps reduced to one number
mostly for clasifiers
"""
model = keras.Sequential([
    #pretrained_base,
    layers.GlobalAvgPool2D(),
    layers.Dense(1, activation='sigmoid'),
])