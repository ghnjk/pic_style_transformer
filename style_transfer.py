#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: style_transfer
@author: jkguo
@create: 2023/2/3
"""
import numpy as np
import tensorflow as tf
from PIL import Image


def gram(tensor: tf.Tensor):
    shape = tensor.shape
    # print(shape)
    channel_count = shape[2]
    n = shape[0] * shape[1]
    tensor = tf.transpose(tensor, perm=(2, 0, 1))
    tensor = tf.reshape(tensor, shape=(channel_count, n))
    return tf.matmul(tensor, tf.transpose(tensor)) / (n * channel_count)


def tf_gram_matrix(input_tensor: tf.Tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


class StyleContentModel(tf.keras.Model):

    def __init__(self, style_layer_names, content_layer_name, use_tf_gram=True):
        super(StyleContentModel, self).__init__()
        self.vgg = self.__build_vgg_layers(
            style_layer_names + content_layer_name
        )
        self.style_layer_names = style_layer_names
        self.content_layer_name = content_layer_name
        self.vgg.trainable = False
        self.use_tf_gram = use_tf_gram
        if self.use_tf_gram:
            self.style_loss_weight = 1e-2
            self.content_loss_wight = 1e4
            self.total_variation_weight = 30.0
        else:
            self.style_loss_weight = 1.0
            self.content_loss_wight = 1e3
            self.total_variation_weight = 30.0

    def call(self, inputs, training=None, mask=None):
        """
        Expects float input in [0,1]
        :param mask:
        :param training:
        :param inputs:
        :return:
        """
        image = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            image
        )
        outputs = self.vgg(preprocessed_input)
        style_outputs = outputs[: len(self.style_layer_names)]
        style_features_dict = {
            style_name: value for style_name, value in zip(
                self.style_layer_names, style_outputs
            )
        }
        content_outputs = outputs[len(self.style_layer_names):]
        if self.use_tf_gram:
            style_outputs = [tf_gram_matrix(style_output) for style_output in style_outputs]
        else:
            style_outputs = [gram(style_output[0]) for style_output in style_outputs]
        content_dict = {
            content_name: value for content_name, value in zip(
                self.content_layer_name, content_outputs
            )
        }
        style_dict = {
            style_name: value for style_name, value in zip(
                self.style_layer_names, style_outputs
            )
        }
        return {
            "content": content_dict,
            "style": style_dict,
            "style_features": style_features_dict
        }

    @staticmethod
    def __build_vgg_layers(output_layer_names):
        """
        根据layer_names获取vgg19的特征model
        :param output_layer_names:
        :return:
        """
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        outputs = [vgg.get_layer(layer_name).output for layer_name in output_layer_names]
        return tf.keras.Model([vgg.input], outputs)

    @staticmethod
    def high_pass_x_y(image: tf.Tensor):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

        return x_var, y_var

    def total_variation_loss(self, image: tf.Tensor):
        x_deltas, y_deltas = self.high_pass_x_y(image)
        return tf.cast(tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas)), dtype=tf.float64)

    def calc_loss(self, gen_image, gen_img_output, style_img_output, content_img_output):
        style_content = gen_img_output["style"]
        content_output = gen_img_output["content"]
        style_target = style_img_output["style"]
        content_target = content_img_output["content"]
        style_loss = tf.add_n([
            tf.reduce_mean(
                (style_content[name] - style_target[name]) ** 2
            ) for name in style_content.keys()
        ])
        style_loss /= len(style_content)
        content_loss = tf.add_n([
            tf.reduce_mean(
                (content_output[name] - content_target[name]) ** 2
            ) for name in content_output.keys()
        ])
        content_loss /= len(content_output)
        total_var_loss = tf.cast(self.total_variation_loss(gen_image), dtype=tf.float32)
        loss = self.style_loss_weight * style_loss + self.content_loss_wight * content_loss + self.total_variation_weight * total_var_loss
        return {
            "loss": loss,
            "style_loss": style_loss,
            "content_loss": content_loss,
            "total_variation_loss": total_var_loss
        }


class StyleTransfer(object):

    def __init__(self, image_height: int = 640, image_width: int = 960):
        self.image_height: int = image_height
        self.image_width: int = image_width
        self.style_layer_names = ['block1_conv1',
                                  'block2_conv1',
                                  'block3_conv1',
                                  'block4_conv1',
                                  'block5_conv1']
        self.content_layer_names = [
            'block5_conv2'
        ]
        self.extractor = StyleContentModel(self.style_layer_names, self.content_layer_names, use_tf_gram=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.99, epsilon=1e-1)

    def load_image(self, image_file_path: str):
        im = Image.open(image_file_path)
        width, height = im.size
        r = self.image_width * 1.0 / self.image_height
        # calc center crop size
        crop_h = height
        crop_w = int(height * r)
        if crop_w > width:
            r = self.image_height * 1.0 / self.image_width
            crop_w = width
            crop_h = int(width * r)
        # center crop image
        left = (width - crop_w) // 2
        right = left + crop_w
        top = (height - crop_h) // 2
        bottom = top + crop_h
        im = im.crop((left, top, right, bottom))
        # im.show()
        # print(f"cropped size {im.size}")
        # resize picture
        im = im.resize((self.image_width, self.image_height))
        # print(f"size {im.size}")
        return np.asarray(im)[:, :, :3] / 255.0

    def transfer(self, content_image, style_image, epoch=10, step_per_epoch=10, gen_image=None):
        style_img_output = self.extractor(np.array([style_image]))
        content_img_output = self.extractor(np.array([content_image]))
        if gen_image is None:
            # gen_image = tf.Variable(tf.random.uniform(minval=0.0, maxval=1.0, shape=np.array([content_image]).shape))
            gen_image = tf.Variable(np.array([content_image]))
        for e in range(epoch):
            losses = {}
            for step in range(step_per_epoch):
                losses = self.train_step(
                    gen_image, style_img_output, content_img_output
                )
            loss = losses["loss"].numpy()
            style_loss = losses["style_loss"].numpy()
            content_loss = losses["content_loss"].numpy()
            total_variation_loss = losses["total_variation_loss"].numpy()
            print(
                "epoch {} step {} loss {:.2f} style_loss {:.2f} content_loss {:.2f} total_variation_loss {:.2f}".format(
                    e + 1,
                    (e + 1) * step_per_epoch,
                    loss,
                    style_loss,
                    content_loss,
                    total_variation_loss
                ))
        return gen_image[0]

    @tf.function
    def train_step(self, gen_image: tf.Variable, style_img_output, content_img_output):
        with tf.GradientTape() as tape:
            outputs = self.extractor(gen_image)
            losses = self.extractor.calc_loss(gen_image, outputs, style_img_output, content_img_output)

        img_grad = tape.gradient(losses["loss"], gen_image)
        self.optimizer.apply_gradients([
            (img_grad, gen_image)
        ])
        gen_image.assign(clip_0_1(gen_image))
        return losses
