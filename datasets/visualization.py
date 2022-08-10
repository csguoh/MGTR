import cv2
import os
import matplotlib.pyplot as plt
import json
from datasets.laeo import convert_xywh2x1y1x2y2

def vis(img, anno):
    img = img.copy()
    item = anno['gt_bboxes']
    for ins in item:
        x1, y1, x2, y2 = convert_xywh2x1y1x2y2(ins['box'],(anno['height'],anno['width']),0)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 64, 0), 3)

    cv2.imshow('show', img)
    if cv2.waitKey(500) == 27:
        cv2.imwrite('./log/{}'.format(anno['file_name'].split('/')[-1]),img)


if __name__ == '__main__':
    annoFile = './data/ava_testScence.json'
    win = cv2.namedWindow('show')
    clean_path ='F:\\AVA_dataset\\frames'
    annotations = [json.loads(l.strip()) for l in open(annoFile, 'r').readlines()]
    for anno in annotations:
        img = cv2.imread(os.path.join(clean_path, anno['file_name']))
        vis(img, anno)
