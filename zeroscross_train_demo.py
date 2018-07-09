import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import yaml

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
from PIL import Image
import mrcnn.model as modellib
from mrcnn import visualize

import matplotlib.pyplot as plt

# switch matplotlib backend for linux
plt.switch_backend("agg")

# Directory to save logs and trained model

iter_num = 0


class ZerosCrossConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "zerocrossline"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 360
    IMAGE_MAX_DIM = 640

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = ZerosCrossConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class ZerosCrossDataset(utils.Dataset):

    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "zerocross")
        # Add images
        for i in range(count):
            filestr = imglist[i].split(".")[0]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "seg_json_dataset/" + filestr + "_json/info.yaml"
            print(dataset_root_path + "seg_json_dataset/" + filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + "seg_json_dataset/" + filestr + "_json/img.png")
            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # overwrite load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        # print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion,
                                       np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("zerocross") != -1:
                labels_form.append("zerocross")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


# base setting
dataset_root_path = '/home/jiangmingchao/seg_train_demo/'
img_floder = dataset_root_path + "height_point_train_data"
mask_floder = dataset_root_path + "mask8"
MODEL_DIR = "/home/jiangmingchao/Mask_RCNN-master/logs/"
COCO_MODEL_PATH = "/home/jiangmingchao/Mask_RCNN-master/mask_rcnn_coco.h5"
imglist = os.listdir(img_floder)
count = len(imglist)

# train and val dataset prepare
dataset_train = ZerosCrossDataset()
dataset_train.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
dataset_train.prepare()

dataset_val = ZerosCrossDataset()
dataset_val.load_shapes(20, img_floder, mask_floder, imglist,dataset_root_path)
dataset_val.prepare()

# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 1)
# for image_id in image_ids:
#    image = dataset_train.load_image(image_id)
#    mask, class_ids = dataset_train.load_mask(image_id)
#    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

print("Create model in training mode")
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

print("load model weights")
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=20,
            layers="all")