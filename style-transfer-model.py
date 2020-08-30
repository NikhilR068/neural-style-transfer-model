
import tensorflow as tf
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import time
import argparse
import warnings
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D,MaxPooling2D
import numpy as np
import imageio
from PIL import Image
from skimage import color
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import convert_all_kernels_in_model
def imread(path, mode="RGB"):
    
    img = np.array(imageio.imread(path, pilmode=mode))
    return img
    

def imresize(img, size, interp='bilinear'):
    if interp == 'bilinear':
        interpolation = Image.BILINEAR
    elif interp == 'bicubic':
        interpolation = Image.BICUBIC
    else:
        interpolation = Image.NEAREST

    
    size = (size[1], size[0])

    if type(img) != Image:
        img = Image.fromarray(img, mode='RGB')

    img = np.array(img.resize(size, interpolation))
    return img
    
    
def imsave(path, img):
    imageio.imwrite(path, img)
    return
    

def fromimage(img, mode='RGB'):
    if mode == 'RGB':
        img = color.lab2rgb(img)
    else:
        img = color.rgb2lab(img)
    return img


def toimage(arr, mode='RGB'):
    return Image.fromarray(arr, mode)

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

TF_19_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


base_image_path = 'winter-wolf.jpg'

syle_image_paths = ['seated-nude.jpg']

result_prefix = 'ti'


img_size = 500

content_weight = 0.025

style_weight = [1.0]
style_scale = 1.0

total_variation_weight = 8.5e-5

num_iter = 10
modelt = 'vgg19'

content_loss_type = 0

rescale_method = 'bilinear'

maintain_aspect_ratio = 'False'

content_layer = 'conv5_2'

init_image = 'content'

min_improvement = 0.0

def str_to_bool(v):
    return v.lower() in ("true", "yes", "t", "1")




style_reference_image_paths = syle_image_paths


style_image_paths = []
for style_image_path in style_reference_image_paths:
    style_image_paths.append(style_image_path)



maintain_aspect_ratio = str_to_bool(maintain_aspect_ratio)


content_weight = content_weight


style_weights = []

if len(style_image_paths) != len(style_weight):
    print("number of style weight and images not equal")

    weight_sum = sum(style_weight) * style_scale
    count = len(style_image_paths)

    for i in range(len(style_image_paths)):
        style_weights.append(weight_sum / count)
else:
    for style_weight in style_weight:
        style_weights.append(style_weight * style_scale)


pooltype = 0

read_mode = "gray" if init_image == "gray" else "color"

img_width = img_height = 0

img_WIDTH = img_HEIGHT = 0
aspect_ratio = 0

assert content_loss_type in [0, 1, 2], "Content Loss Type must be one of 0, 1 or 2"



def preprocess_image(image_path, load_dims=False, read_mode="color"):
    global img_width, img_height, img_WIDTH, img_HEIGHT, aspect_ratio

    mode = "RGB" if read_mode == "color" else "L"
    img = imread(image_path, mode=mode)  

    if mode == "L":
        
        temp = np.zeros(img.shape + (3,), dtype=np.uint8)
        temp[:, :, 0] = img
        temp[:, :, 1] = img.copy()
        temp[:, :, 2] = img.copy()

        img = temp

    if load_dims:
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = float(img_HEIGHT) / img_WIDTH

        img_width = img_size
        if maintain_aspect_ratio:
            img_height = int(img_width * aspect_ratio)
        else:
            img_height = img_size

    img = imresize(img, (img_width, img_height)).astype('float32')

    # RGB -> BGR
    img = img[:, :, ::-1]

    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    if K.image_data_format() == "channels_first":
        img = img.transpose((2, 0, 1)).astype('float32')

    img = np.expand_dims(img, axis=0)
    return img


def deprocess_image(x):
    if K.image_data_format() == "channels_first":
        x = x.reshape((3, img_width, img_height))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_width, img_height, 3))

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # BGR -> RGB
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

def pooling_func(x):
    if pooltype == 1:
        return AveragePooling2D((2, 2), strides=(2, 2))(x)
    else:
        return MaxPooling2D((2, 2), strides=(2, 2))(x)



base_image = K.variable(preprocess_image(base_image_path, True, read_mode=read_mode))

style_reference_images = []
for style_path in style_image_paths:
    style_reference_images.append(K.variable(preprocess_image(style_path)))




combination_image = K.placeholder((1, img_width, img_height, 3))

image_tensors = [base_image]
for style_image_tensor in style_reference_images:
    image_tensors.append(style_image_tensor)
image_tensors.append(combination_image)

nb_tensors = len(image_tensors)
nb_style_images = nb_tensors - 2 
input_tensor = K.concatenate(image_tensors, axis=0)



shape = (nb_tensors, img_width, img_height, 3)

ip = Input(tensor=input_tensor, batch_shape=shape)


x = Conv2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(ip)
x = Conv2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(x)
x = pooling_func(x)

x = Conv2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(x)
x = pooling_func(x)

x = Conv2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(x)
x = pooling_func(x)

x = Conv2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(x)
x = pooling_func(x)

x = Conv2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(x)
x = pooling_func(x)

model = Model(ip, x)


if model == "vgg19":
    weights = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_19_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
else:
    weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')

model.load_weights(weights)

if K.backend() == 'tensorflow' and K.image_data_format() == "channels_first":
    convert_all_kernels_in_model(model)

print('Model loaded.')
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
shape_dict = dict([(layer.name, layer.output_shape) for layer in model.layers])

def gram_matrix(input_tensor):
    assert K.ndim(input_tensor) == 3
    
    features = K.batch_flatten(K.permute_dimensions(input_tensor, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



def style_loss(style, combination,nb_channels=None):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_width * img_height
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def content_loss(base, combination):
    channel_dim = 0 if K.image_data_format() == "channels_first" else -1

    try:
        channels = K.int_shape(base)[channel_dim]
    except TypeError:
        channels = K.shape(base)[channel_dim]
    size = img_width * img_height

    if content_loss_type == 1:
        multiplier = 1. / (2. * (channels ** 0.5) * (size ** 0.5))
    elif content_loss_type == 2:
        multiplier = 1. / (channels * size)
    else:
        multiplier = 1.

    return multiplier * K.sum(K.square(combination - base))
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == "channels_first":
        a = K.square(x[:, :, :img_width - 1, :img_height - 1] - x[:, :, 1:, :img_height - 1])
        b = K.square(x[:, :, :img_width - 1, :img_height - 1] - x[:, :, :img_width - 1, 1:])
    else:
        a = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, 1:, :img_height - 1, :])
        b = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, :img_width - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


loss = K.variable(0.)
layer_features = outputs_dict[content_layer]  # 'conv5_2' or 'conv4_2'
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[nb_tensors - 1, :, :, :]
loss = loss + content_weight * content_loss(base_image_features,
                                      combination_features)

channel_index = 1 if K.image_data_format() == "channels_first" else -1

feature_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    shape = shape_dict[layer_name]
    combination_features = layer_features[nb_tensors - 1, :, :, :]

    style_reference_features = layer_features[1:nb_tensors - 1, :, :, :]
    sl = []
    for j in range(nb_style_images):
        sl.append(style_loss(style_reference_features[j], combination_features, shape))

    for j in range(nb_style_images):
        loss = loss + (style_weights[j] / len(feature_layers)) * sl[j]

loss = loss + total_variation_weight * total_variation_loss(combination_image)


grads = K.gradients(loss, combination_image)

outputs = [loss]
if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)


def eval_loss_and_grads(x):
    if K.image_data_format() == "channels_first":
        x = x.reshape((1, 3, img_width, img_height))
    else:
        x = x.reshape((1, img_width, img_height, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()

if "content" in init_image or "gray" in init_image:
    x = preprocess_image(base_image_path, True, read_mode=read_mode)
elif "noise" in init_image:
    x = np.random.uniform(0, 255, (1, img_width, img_height, 3)) - 128.

    if K.image_data_format() == "channels_first":
        x = x.transpose((0, 3, 1, 2))
else:
    print("Using initial image : ", init_image)
    x = preprocess_image(init_image, read_mode=read_mode)


num_iter = num_iter
prev_min_val = -1

improvement_threshold = float(min_improvement)

for i in range(num_iter):
    print("Starting iteration %d of %d" % ((i + 1), num_iter))
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)

    if prev_min_val == -1:
        prev_min_val = min_val

    improvement = (prev_min_val - min_val) / prev_min_val * 100

    print('Current loss value:', min_val, " Improvement : %0.3f" % improvement, "%")
    prev_min_val = min_val
    # save generated image
    img = deprocess_image(x.copy())

    
    
    img_ht = int(img_width * aspect_ratio)
    img = imresize(img, (img_width, img_ht), interp=rescale_method)

    fname = result_prefix + '_%d.png' % (i + 1)
    imsave(fname, img)
    
    print('Image saved as', fname)

    if improvement_threshold is not 0.0:
        if improvement < improvement_threshold and improvement is not 0.0:
            print("Improvement (%f) is less than improvement threshold (%f). Early stopping script." % (
                improvement, improvement_threshold))
            exit()
