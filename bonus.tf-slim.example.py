from tensorflow.contrib import slim

import sys
# /home/crapas/tf_src 위치에서 git clone을 실행한 경우의 디렉토리 설정
sys.path.append("/Users/gamgoon/git/models/research/slim")

from datasets import dataset_utils
import tensorflow as tf
from urllib.request import urlopen
from nets import vgg
from preprocessing import vgg_preprocessing
import os

# 팁을 하나 추가하자면, 아래의 예제코드는 vgg-16모델을 내려받아 압축 해제한 후 실행해야 한다. 실행할 때마다 내려받는 게 단점이긴 하지만
# 다음과 같은 코드를 삽입하면 미리 내려받지 않고 테스트 가능하다.
vgg_url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
# vgg-16 모델을 내려받아 압축을 해제한 위치(이 경로에 vgg_16.ckpt 파일이 있어야 함)
target_dir = '/Users/gamgoon/git/learning.tensorflow/checkpoints'
if not tf.gfile.Exists(target_dir):
    tf.gfile.MakeDirs(target_dir)
dataset_utils.download_and_uncompress_tarball(vgg_url, target_dir)

# target_dir = '/Users/gamgoon/git/learning.tensorflow/checkpoints'
url = ("http://54.68.5.226/car.jpg")
im_as_string = urlopen(url).read()
image = tf.image.decode_jpeg(im_as_string, channels=3)

image_size = vgg.vgg_16.default_image_size

processed_im = vgg_preprocessing.preprocess_image(image,
                                                  image_size,
                                                  image_size,
                                                  is_training=False)
processed_images = tf.expand_dims(processed_im, 0)                                                  

with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, _ = vgg.vgg_16(processed_images,
                           num_classes=1000,
                           is_training=False)

probabilities = tf.nn.softmax(logits)

def vgg_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                         activation_fn=tf.nn.relu,
                         weights_regularizer=slim.l2_regularizer(weight_decay),
                         baiases_initializer=tf.zeros_initializer):
        with slim.arg_scope([slim.conv2d], padding='SAME') as args_sc:
            return args_sc

load_vars = slim.assign_from_checkpoint_fn(
    os.path.join(target_dir, 'vgg_16.ckpt'),
    slim.get_model_variables('vgg_16'))

from datasets import imagenet

with tf.Session() as sess:
    load_vars(sess)
    network_input, probabilities = sess.run([processed_images,
                                             probabilities])
    probabilities = probabilities[0, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                                key=lambda x:x[1])]

    names_ = imagenet.create_readable_names_for_imagenet_labels()

    for i in range(5):
        index = sorted_inds[i]
        print('Class: ' + names_[index+1]
            + ' |prob: ' + str(probabilities[index]))

