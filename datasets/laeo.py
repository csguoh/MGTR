from torchvision.datasets.vision import VisionDataset
import torchvision
import torch
import numpy as np
import json
import cv2
import random
import PIL
import torchvision.transforms as T
import torchvision.transforms.functional as F
from util.box_ops import box_xyxy_to_cxcywh
from PIL import Image
from matplotlib import pyplot as plt


def check_bbox(img1, target):
    img = np.array(img1.copy())
    for bix in range(0, len(target['person1_boxes'])):
        cv2.rectangle(img, (target['person1_boxes'][bix][0].int(),
                            target['person1_boxes'][bix][1].int()),
                      (target['person1_boxes'][bix][2].int(),
                       target['person1_boxes'][bix][3].int()),
                      (255, 64, 0), 3)
        cv2.rectangle(img, (target['person2_boxes'][bix][0].int(), target['person2_boxes'][bix][1].int()),
                      (target['person2_boxes'][bix][2].int(), target['person2_boxes'][bix][3].int()),
                      (255, 64, 0), 3)
        if target['action_labels'][bix] == 1:
            p1 = (target['person1_boxes'][bix][0].int(), target['person1_boxes'][bix][1].int())
            p2 = (target['person2_boxes'][bix][2].int(), target['person2_boxes'][bix][3].int())
            cv2.line(img, p1, p2, (0, 255, 0), 3)

    cv2.imshow('img', img)
    cv2.waitKey(30)


def convert_xywh2x1y1x2y2(box, shape, flip):
    ih, iw = shape[:2]
    x, y, w, h = box
    if flip == 1:
        x1_org = x
        x2_org = x + w - 1
        x2 = iw - 1 - x1_org
        x1 = iw - 1 - x2_org
    else:
        x1 = x
        x2 = x + w - 1
    x1 = max(x1, 0)
    x2 = min(x2, iw - 1)
    y1 = max(y, 0)
    y2 = min(y + h - 1, ih - 1)
    return [x1, y1, x2, y2]


def get_det_annotation_from_annotator(item, shape, flip, gt_size_min=1):
    total_boxes, gt_boxes, ignored_boxes = [], [], []
    for annot in item['gt_bboxes']:
        box = convert_xywh2x1y1x2y2(annot['box'], shape, flip)
        x1, y1, x2, y2 = box
        cls_id = 1
        total_boxes.append([x1, y1, x2, y2, cls_id, ])
        if annot['tag'] not in [1]:
            continue
        if annot.get('extra', {}).get('ignore', 0) == 1:
            ignored_boxes.append(box)
            continue
        if (x2 - x1 + 1) * (y2 - y1 + 1) < gt_size_min ** 2:
            ignored_boxes.append(box)
            continue
        if x2 <= x1 or y2 <= y1:
            ignored_boxes.append(box)
            continue
        gt_boxes.append([x1, y1, x2, y2, cls_id, ])
    return gt_boxes, ignored_boxes, total_boxes


def get_interaction_box(person1_box, person2_box, laeo_id):
    hx1, hy1, hx2, hy2, hid = person1_box
    ox1, oy1, ox2, oy2, oid = person2_box
    xx1, yy1, xx2, yy2 = min(hx1, ox1), min(hy1, oy1), max(hx2, ox2), max(hy2, oy2)
    return [xx1, yy1, xx2, yy2, laeo_id]


def xyxy_to_cxcywh(box):
    x0, y0, x1, y1, cid = box
    return [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0), cid]


def get_laeo_annotation_from_annotator(item, total_boxes, scale):
    person1_boxes, person2_boxes, action_boxes = [], [], []
    person1_labels, person2_labels, action_labels = [], [], []
    img_hh, img_ww = item['height'], item['width']
    for laeo_pair in item.get('laeo', []):
        x1, y1, x2, y2, cls_id = list(map(int, total_boxes[laeo_pair['person_1']]))
        person1_box = x1 // scale, y1 // scale, x2 // scale, y2 // scale, cls_id
        if cls_id == -1 or x1 >= x2 or y1 >= y2:
            continue
        x1, y1, x2, y2, cls_id = list(map(int, total_boxes[laeo_pair['person_2']]))
        person2_box = x1 // scale, y1 // scale, x2 // scale, y2 // scale, cls_id
        if cls_id == -1 or x1 >= x2 or y1 >= y2:
            continue
        laeo_id = laeo_pair['interaction']
        laeo_box = get_interaction_box(person1_box=person1_box, person2_box=person2_box, laeo_id=laeo_id)

        person1_boxes.append(person1_box[0:4])
        person2_boxes.append(person2_box[0:4])
        action_boxes.append(laeo_box[0:4])
        person1_labels.append(person1_box[4])
        person2_labels.append(person2_box[4])
        action_labels.append(laeo_box[4])
    return dict(
        person1_boxes=torch.from_numpy(np.array(person1_boxes).astype(np.float32)),
        person1_labels=torch.from_numpy(np.array(person1_labels)),
        person2_boxes=torch.from_numpy(np.array(person2_boxes).astype(np.float32)),
        person2_labels=torch.from_numpy(np.array(person2_labels)),
        action_boxes=torch.from_numpy(np.array(action_boxes).astype(np.float32)),
        action_labels=torch.from_numpy(np.array(action_labels)),
        image_id=item['file_name'],
        org_size=torch.as_tensor([int(img_hh), int(img_ww)]),
    )


def parse_one_gt_line(gt_line, scale=1):
    item = json.loads(gt_line)
    img_name = item['file_name']
    img_shape = item['height'], item['width']
    gt_boxes, ignored_boxes, total_boxes = get_det_annotation_from_annotator(item, img_shape, flip=0)
    interaction_boxes = get_laeo_annotation_from_annotator(item, total_boxes, scale)
    return dict(image_id=img_name, annotations=interaction_boxes)


def hflip(image, target, image_set='train'):
    flipped_image = F.hflip(image)
    target = target.copy()
    if image_set in ['test']:
        return flipped_image, target

    w, h = image.size
    if "person1_boxes" in target:
        boxes = target["person1_boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["person1_boxes"] = boxes
    if "person2_boxes" in target:
        boxes = target["person2_boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["person2_boxes"] = boxes
    if "action_boxes" in target:
        boxes = target["action_boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["action_boxes"] = boxes
    return flipped_image, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target, image_set='train'):
        if random.random() < self.p:
            return hflip(img, target, image_set)
        return img, target


class RandomAdjustImage(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target, image_set='train'):
        if random.random() < self.p:
            img = F.adjust_brightness(img, random.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
        if random.random() < self.p:
            img = F.adjust_contrast(img, random.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
        return img, target


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target, image_set='train'):
        if random.random() < self.p:
            return self.transforms1(img, target, image_set)
        return self.transforms2(img, target, image_set)


def resize(image, target, size, max_size=None, image_set='train'):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return h, w
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return oh, ow

    rescale_size = get_size_with_aspect_ratio(image_size=image.size, size=size, max_size=max_size)
    rescaled_image = F.resize(image, rescale_size)

    if target is None:
        return rescaled_image, None
    target = target.copy()
    if image_set in ['test']:
        return rescaled_image, target

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    if "person1_boxes" in target:
        boxes = target["person1_boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["person1_boxes"] = scaled_boxes
    if "person2_boxes" in target:
        boxes = target["person2_boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["person2_boxes"] = scaled_boxes
    if "action_boxes" in target:
        boxes = target["action_boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["action_boxes"] = scaled_boxes
    return rescaled_image, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None, image_set='train'):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size, image_set)


def crop(image, org_target, region, image_set='train'):
    cropped_image = F.crop(image, *region)
    target = org_target.copy()
    if image_set in ['test']:
        return cropped_image, target

    i, j, h, w = region
    fields = ["person1_labels", "person2_labels", "action_labels"]

    if "person1_boxes" in target:
        boxes = target["person1_boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["person1_boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("person1_boxes")
    if "person2_boxes" in target:
        boxes = target["person2_boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["person2_boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("person2_boxes")
    if "action_boxes" in target:
        boxes = target["action_boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["action_boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("action_boxes")

    # remove elements for which the boxes or masks that have zero area
    if "person1_boxes" in target and "person2_boxes" in target:
        cropped_boxes = target['person1_boxes'].reshape(-1, 2, 2)
        keep1 = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        cropped_boxes = target['person2_boxes'].reshape(-1, 2, 2)
        keep2 = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        keep = keep1 * keep2
        if keep.any().sum() == 0:
            return image, org_target
        for field in fields:
            target[field] = target[field][keep]
    return cropped_image, target


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict, image_set='train'):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, (h, w))
        return crop(img, target, region, image_set)


class ToTensor(object):
    def __call__(self, img, target, image_set='train'):
        return torchvision.transforms.functional.to_tensor(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, image_set='train'):
        image = torchvision.transforms.functional.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        if image_set in ['test']:
            return image, target
        h, w = image.shape[-2:]
        if "person1_boxes" in target:
            boxes = target["person1_boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["person1_boxes"] = boxes
        if "person2_boxes" in target:
            boxes = target["person2_boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["person2_boxes"] = boxes
        if "action_boxes" in target:
            boxes = target["action_boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["action_boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, image_set='train'):
        for t in self.transforms:
            image, target = t(image, target, image_set)
        return image, target


def make_laeo_transforms(image_set, test_scale=-1):
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if image_set == 'train':
        return Compose([
            RandomHorizontalFlip(),
            RandomAdjustImage(),
            RandomSelect(
                RandomResize(scales, max_size=1333),
                Compose([
                    RandomResize([400, 500, 600]),
                    RandomSizeCrop(384, 600),
                    RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    if image_set == 'test':
        if test_scale == -1:
            return Compose([
                normalize,
            ])
        assert 400 <= test_scale <= 800, test_scale
        return Compose([
            RandomResize([test_scale], max_size=1333),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')


class  MutualGazeDetection(VisionDataset):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None, image_set='train'):
        assert image_set in ['train', 'test'], image_set
        self.image_set = image_set
        super( MutualGazeDetection, self).__init__(root, transforms, transform, target_transform)
        annotations = [parse_one_gt_line(l.strip()) for l in open(annFile, 'r').readlines()]
        if self.image_set in ['train']:
            self.annotations = [a for a in annotations if len(a['annotations']['action_labels']) > 0]
        else:
            self.annotations = annotations
        self.transforms = transforms

    def __getitem__(self, index):
        ann = self.annotations[index]
        img_name = ann['image_id']
        target = ann['annotations']
        img_path = 'E:/uco-laeo/ucolaeodb/frames/' + img_name
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = Image.fromarray(img[:, :, ::-1]).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target, self.image_set)
        return img, target

    def __len__(self):
        return len(self.annotations)


def build(image_set, args, test_scale=-1):
    assert image_set in ['train', 'test'], image_set
    assert args.data_name in ['ava','uco'], args.data_name
    if args.data_name=='uco':
        if image_set == 'train':
            annotation_file = './data/uco_trainScence.json'
        else:
            annotation_file = './data/uco_testScence.json'
    elif args.data_name == 'ava':
        if image_set == 'train':
            annotation_file = './data/ava_trainScence.json'
        else:
            annotation_file = './data/ava_testScence.json'

    dataset = MutualGazeDetection(root='./data', annFile=annotation_file,
                           transforms=make_laeo_transforms(image_set, test_scale), image_set=image_set)
    return dataset



if __name__ == '__main__':
    build('train')
