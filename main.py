import numpy as np
from PIL import Image
import time
import cv2
import tensorflow as tf
from keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
import IPython.display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False


#МАСКА НА СИНИЙ
# bottom = (100, 50, 50)
# top = (155, 255, 255)

#МАСКА НА КРАСНЫЙ
# bottom = (0, 50, 50)
# top = (179, 255, 255)

#МАСКА НА ЗЕЛЕНЫЙ
# bottom = (50, 50, 50)
# top = (70, 255, 255)

#МАСКА НА ЖЕЛТЫЙ
# bottom = (10, 50, 50)
# top = (40, 255, 255)

#МАСКА НА БЕЛЫЙ
#bottom = (0, 0, 200)
#top = (40, 40, 255)

#МАСКА НА ЧЕРНЫЙ
#bottom = (2, 0, 0)
#top = (120, 50, 60)


#ЗАПРОСИТЬ ЕЩЕ МАШИНОК
#НАЙТИ ЕЩЕ СТИЛЕЙ


# load image
img = cv2.imread("images/content/test_content_13.jpg")
style_path = 'images/style/test_style_4.jpg'

max_dim = 512
iterations = 100


# mask
hsv_car = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
bottom = (2, 0, 0)
top = (120, 50, 60)
mask = cv2.inRange(hsv_car, bottom, top)
corners = cv2.goodFeaturesToTrack(mask, 200, 0.01, 10)
corners = np.int0(corners)

left = np.array([100000, 100000])
right = np.array([-1, -1])


for i in corners:
    x, y = i.ravel()

    if x <= left[0]:
        left[0] = x
    if x >= right[0]:
        right[0] = x

    if y <= left[1]:
        left[1] = y
    if y >= right[1]:
        right[1] = y



img2 = cv2.bitwise_and(img, img, mask = mask)
img3 = img[left[1] - 10:right[1] + 60, left[0] - 10:right[0] + 10]
img2 = img2[left[1] - 10:right[1] + 60, left[0] - 10:right[0] + 10]
mask = mask[left[1] - 10:right[1] + 60, left[0] - 10:right[0] + 10]

invMask = cv2.bitwise_not(mask)


cv2.namedWindow('Image', cv2.WINDOW_KEEPRATIO)
cv2.imshow('Image', img)
cv2.resizeWindow('Image', 200, 200)

cv2.namedWindow('Image_2', cv2.WINDOW_KEEPRATIO)
cv2.imshow('Image_2', img2)
cv2.imwrite('images/extra/cropped_content.jpg', img2)
cv2.resizeWindow('Image_2', 200, 200)

cv2.namedWindow('Mask', cv2.WINDOW_KEEPRATIO)
cv2.imshow('Mask', mask)
cv2.imwrite('images/masks/mask.jpg', mask)
cv2.resizeWindow('Mask', 200, 200)

cv2.namedWindow('InvMask', cv2.WINDOW_KEEPRATIO)
cv2.imshow('InvMask', invMask)
cv2.imwrite('images/masks/invMask.jpg', invMask)
cv2.resizeWindow('InvMask', 200, 200)

#cv2.waitKey(0)

first_res = cv2.bitwise_and(img3, img3, mask = invMask)
cv2.imwrite('images/extra/first_res.jpg', first_res)




croppedTmp = Image.open("images/extra/cropped_content.jpg")
long = max(croppedTmp.size)
scale = max_dim / long
croppedTmp = croppedTmp.resize((round(croppedTmp.size[0] * scale), round(croppedTmp.size[1] * scale)), Image.ANTIALIAS)
croppedTmp.save("images/extra/cropped_content.jpg", quality=95)




content_path = 'images/extra/cropped_content.jpg'


def load_img(path_to_img):
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)

    img = kp_image.img_to_array(img)

    img = np.expand_dims(img, axis=0)
    return img

def imshow(img, title=None):
  out = np.squeeze(img, axis=0)
  out = out.astype('uint8')
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out)

plt.figure(figsize=(10,10))


content = load_img(content_path)

style = load_img(style_path)


plt.subplot(1, 2, 1)
imshow(content, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style, 'Style Image')
plt.show()

def load_and_process_img(path_to_img):
  img = load_img(path_to_img)
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

content_layers = ['block5_conv2']


style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def get_model():

    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs

    return models.Model(vgg.input, model_outputs)

def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):

    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):

    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))


def get_feature_representations(model, content_path, style_path):

    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    style_outputs = model(style_image)
    content_outputs = model(content_image)

    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):

    style_weight, content_weight = loss_weights

    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    weight_per_style_layer = 1 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    weight_per_content_layer = 1 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    loss = style_score + content_score
    return loss, style_score, content_score

def compute_grads(cfg):
  with tf.GradientTape() as tape:
    all_loss = compute_loss(**cfg)

  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss


def run_style_transfer(content_path,
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):

    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    opt = tf.optimizers.Adam(learning_rate=1, beta_1=0.99, epsilon=1e-1)

    best_loss, best_img = float('inf'), None

    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    num_rows = 2
    num_cols = 5
    display_interval = num_iterations / (num_rows * num_cols)
    start_time = time.time()
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time()

        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

        if i % display_interval == 0:
            start_time = time.time()

            plot_img = init_image.numpy()
            plot_img = deprocess_img(plot_img)
            imgs.append(plot_img)
            IPython.display.clear_output(wait=True)
            IPython.display.display_png(Image.fromarray(plot_img))
            print('Iteration: {}'.format(i))
            print('Total loss: {:.4e}, '
                  'style loss: {:.4e}, '
                  'content loss: {:.4e}, '
                  'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    IPython.display.clear_output(wait=True)
    plt.figure(figsize=(14, 4))
    for i, img in enumerate(imgs):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    return best_img, best_loss

best, best_loss = run_style_transfer(content_path,
                                     style_path, num_iterations=iterations)



def show_results(best_img, content_path, style_path, show_large_final=True):
  plt.figure(figsize=(10, 5))
  content = load_img(content_path)
  style = load_img(style_path)

  plt.subplot(1, 2, 1)
  imshow(content, 'Content Image')

  plt.subplot(1, 2, 2)
  imshow(style, 'Style Image')

  if show_large_final:
    plt.figure(figsize=(10, 10))

    plt.imshow(best_img)
    plt.title('Output Image')
    plt.show()

show_results(best, content_path, style_path)

result = Image.fromarray(best)
result.save('images/extra/result.jpg')

mask_im = Image.open('images/masks/mask.jpg')
temp = Image.open('images/extra/first_res.jpg')

long = max(temp.size)
scale = max_dim / long
temp = temp.resize((round(temp.size[0] * scale), round(temp.size[1] * scale)), Image.ANTIALIAS)
temp.save('images/extra/first_res_fixed.jpg', quality=95)

long = max(mask_im.size)
scale = max_dim / long
mask_im = mask_im.resize((round(mask_im.size[0] * scale), round(mask_im.size[1] * scale)), Image.ANTIALIAS)
mask_im.save('images/masks/mask_fixed.jpg', quality=95)

res_im = cv2.imread('images/extra/result.jpg')
mask_fixed = cv2.imread('images/masks/mask_fixed.jpg', cv2.IMREAD_GRAYSCALE)


res_im = cv2.bitwise_and(res_im, res_im, mask = mask_fixed)


temp = cv2.imread("images/extra/first_res_fixed.jpg")

dst = cv2.add(res_im, temp)
cv2.imwrite('images/results/final_10.jpg', dst)



