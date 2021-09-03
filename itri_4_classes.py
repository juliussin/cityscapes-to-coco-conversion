# This file heavily borrows from https://github.com/facebookresearch/Detectron/tree/master/tools

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

# Image processing
# Check if PIL is actually Pillow as expected
try:
    from PIL import PILLOW_VERSION
except:
    print("Please install the module 'Pillow' for image processing, e.g.")
    print("pip install pillow")
    sys.exit(-1)

try:
    import PIL.Image     as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)

import argparse
import json
import os
import cv2
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from utils.instance_class import *
from utils.labels import *

# def findContours(*args, **kwargs):
#     """
#     Wraps cv2.findContours to maintain compatiblity between versions
#     3 and 4
#     Returns:
#         contours, hierarchy
#     """
#     if cv2.__version__.startswith('4'):
#         contours, hierarchy = cv2.findContours(*args, **kwargs)
#     elif cv2.__version__.startswith('3'):
#         _, contours, hierarchy = cv2.findContours(*args, **kwargs)
#     else:
#         raise AssertionError(
#             'cv2 must be either version 3 or 4 to call this method')

#     return contours, hierarchy

# def instances2dict_with_polygons(imageFileList, verbose=False):
#     imgCount     = 0
#     instanceDict = {}

#     if not isinstance(imageFileList, list):
#         imageFileList = [imageFileList]

#     if verbose:
#         print("Processing {} images...".format(len(imageFileList)))

#     for imageFileName in imageFileList:
#         # Load image
#         img = Image.open(imageFileName)

#         # Image as numpy array
#         imgNp = np.array(img)

#         # Initialize label categories
#         instances = {}
#         for label in labels:
#             instances[label.name] = []

#         # Loop through all instance ids in instance image
#         for instanceId in np.unique(imgNp):
#             if instanceId < 1000:
#                 continue
#             instanceObj = Instance(imgNp, instanceId)
#             instanceObj_dict = instanceObj.toDict()

#             if id2label[instanceObj.labelID].hasInstances:
#                 mask = (imgNp == instanceId).astype(np.uint8)
#                 contour, hier = findContours(
#                     mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#                 polygons = [c.reshape(-1).tolist() for c in contour]
#                 instanceObj_dict['contours'] = polygons

#             instances[id2label[instanceObj.labelID].name].append(instanceObj_dict)

#         instanceDict[imageFileName] = instances
#         imgCount += 1

#         if verbose:
#             print("\rImages Processed: {}".format(imgCount), end=' ')
#             sys.stdout.flush()

#     if verbose:
#         print("")

#     return instanceDict

# def poly_to_box(poly):
#     """Convert a polygon into a tight bounding box."""
#     x0 = min(min(p[::2]) for p in poly)
#     x1 = max(max(p[::2]) for p in poly)
#     y0 = min(min(p[1::2]) for p in poly)
#     y1 = max(max(p[1::2]) for p in poly)
#     box_from_poly = [x0, y0, x1, y1]
#     return box_from_poly

def read_txt(file):
    data = []
    with open(file, 'r') as f:
        lines = f.readlines()
    if len(lines) <= 2: return None, False
    for x in range(len(lines) - 1):
        data.append(lines[x+1][:-1].strip().split())
    return data, True

def xyxy_to_xywh(xyxy_box):
    xmin, ymin, xmax, ymax = xyxy_box
    TO_REMOVE = 1
    xywh_box = (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE)
    return xywh_box

def convert_itri_instance_only(data_dir, out_dir, verbose=False):
    """Convert from ITRI format to COCO format"""
    sets = [
        'leftImg8bit/train',
        'leftImg8bit/val'
    ]

    # ann_dirs = [
    #     'gtFine/train',
    #     'gtFine/val',
    # ]
    ann_dir = 'TXT'
    img_dir = 'RGB'

    split_files = {
        'SPLIT/train.txt',
        'SPLIT/val.txt',
        'SPLIT/test.txt'
    }

    json_name = 'itri_%s.json'
    # polygon_json_file_ending = '_polygons.json'
    annotation_file_ending = '.txt'
    image_file_ending = '.jpg'
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {
        1: 'pedestrian',
        2: 'rider',
        3: 'two-wheels',
        4: 'four-wheels',
    }
    inv_category_dict = {v: k for k, v in category_dict.items()}

    # for data_set, ann_dir in zip(sets, ann_dirs):
    for split_file in split_files:
        print('Starting %s' % split_file)
        with open(split_file, 'r') as f:
            file_list = f.readlines()

        ann_dict = {}
        images = []
        annotations = []

        for fname in file_list:
            # Read txt
            labels = read_txt(os.path.join(data_dir, ann_dir, fname + annotation_file_ending))
            image = {}
            image['id'] = img_id
            img_id += 1
            img_path = os.path.join(data_dir, img_dir, fname + image_file_ending)
            image_shape = plt.imread(img_path).shape
            image['width'] = image_shape[0]
            image['height'] = image_shape[1]
            image['file_name'] = os.path.join(img_dir, fname + image_file_ending)
            images.append(image)

            for label in labels:
                ann = {}
                ann['id'] = ann_id
                ann_id += 1
                ann['image_id'] = image['id']

                ann['category_id'] = inv_category_dict[int(label[0])]
                ann['iscrowd'] = 0

                xywh_box = xyxy_to_xywh(label[1:])
                ann['bbox'] = xywh_box

                annotations.append(ann)

        # for root, _, files in os.walk(os.path.join(data_dir, ann_dir)):
        #     for filename in files if verbose else tqdm(files, desc=os.path.split(root)[1]):
        #         if filename.endswith(annotation_file_ending):
                    
        #             json_ann = json.load(open(os.path.join(root, filename)))

        #             image = {}
        #             image['id'] = img_id
        #             img_id += 1
        #             image['width'] = json_ann['imgWidth']
        #             image['height'] = json_ann['imgHeight']
        #             image['file_name'] = os.path.join("leftImg8bit",
        #                                               data_set.split("/")[-1],
        #                                               filename.split('_')[0],
        #                                               filename.replace("_gtFine_polygons.json", '_leftImg8bit.png'))
        #             image['seg_file_name'] = filename.replace("_polygons.json", "_instanceIds.png")
        #             images.append(image)

        #             fullname = os.path.join(root, image['seg_file_name'])
        #             objects = instances2dict_with_polygons([fullname], verbose=False)[fullname]

        #             for object_cls in objects:
        #                 if object_cls not in category_instancesonly:
        #                     continue  # skip non-instance categories

        #                 for obj in objects[object_cls]:
        #                     if obj['contours'] == []:
        #                         if verbose: print('Warning: empty contours.')
        #                         continue  # skip non-instance categories

        #                     len_p = [len(p) for p in obj['contours']]
        #                     if min(len_p) <= 4:
        #                         if verbose: print('Warning: invalid contours.')
        #                         continue  # skip non-instance categories

        #                     ann = {}
        #                     ann['id'] = ann_id
        #                     ann_id += 1
        #                     ann['image_id'] = image['id']
        #                     # ann['segmentation'] = obj['contours']
        #                     new_object_cls = category_mapping[object_cls]
        #                     # if new_object_cls not in category_dict:
        #                     #     category_dict[new_object_cls] = cat_id
        #                     #     cat_id += 1
        #                     ann['category_id'] = category_dict[new_object_cls]
        #                     ann['iscrowd'] = 0
        #                     ann['area'] = obj['pixelCount']

        #                     #xyxy_box = poly_to_box(ann['segmentation'])
        #                     xyxy_box = poly_to_box(obj['contours'])
        #                     xywh_box = xyxy_to_xywh(xyxy_box)
        #                     ann['bbox'] = xywh_box

        #                     annotations.append(ann)

        ann_dict['images'] = images
        categories = [{"id": idx, "name": category_dict[idx]} for idx in category_dict]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Split: %s" % split_file)
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        if not os.path.exists(os.path.abspath(out_dir)):
            os.makedirs(os.path.abspath(out_dir))
        with open(os.path.join(out_dir, json_name % ann_dir.replace("/", "_")), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('--dataset', help="itri", default='itri', type=str)
    parser.add_argument('--outdir', help="output dir for json files", default='data/itri/annotations', type=str)
    parser.add_argument('--datadir', help="data dir for annotations to be converted", default="data/itri", type=str)
    parser.add_argument('--verbose', help="verbose", action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "itri":
        convert_itri_instance_only(args.datadir, args.outdir, verbose=args.verbose)
    else:
        print("Dataset not supported: %s" % args.dataset)
