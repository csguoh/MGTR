import json
from operator import mod
from matplotlib.pyplot import annotate
import numpy as np
import torch
import os
import random
from PIL import Image

random.seed(1234)
np.random.seed(1234)

PERSON = 1
IS_LAEO = 1
IS_NOT_LAEO = 0
save_path = '/home/edward_krucal/Downloads/Datasets/uco-laeo/tools'
frame_path = '/home/edward_krucal/Downloads/Datasets/uco-laeo/frames'
anno_path = '/home/edward_krucal/Downloads/Datasets/uco-laeo/annotations'
counter_path = '/home/edward_krucal/Downloads/Datasets/uco-laeo/counter'

trainScence = ['twd14', 'sv08', 'mr09', 'got14', 'negatives08', 'negatives04', 'negatives29', 'negatives24', 'sv15',
               'got03', 'sv07', 'sv24', 'twd09', 'got10', 'sv23', 'got20', 'negatives15', 'got02', 'twd29',
               'got13', 'sv21', 'negatives02', 'got04', 'got19', 'negatives05', 'negatives09', 'sv05', 'mr18',
               'negatives28', 'twd22', 'got11', 'mr16', 'negatives06', 'twd23', 'twd02', 'got15', 'twd08', 'twd01',
               'sv01', 'negatives26', 'got01', 'twd21', 'twd16', 'got17', 'mr21', 'negatives17', 'negatives27',
               'twd15', 'got12', 'sv09', 'negatives18', 'twd18', 'sv22', 'got07', 'mr07', 'negatives22', 'sv26',
               'twd11', 'mr11', 'sv27', 'negatives23', 'twd10', 'twd17', 'twd12', 'negatives10', 'mr19', 'twd19',
               'sv25', 'negatives14', 'got23', 'twd20', 'mr17', 'negatives12', 'sv16', 'got18', 'sv13', 'got21',
               'negatives03', 'sv20', 'sv14', 'mr01', 'sv12', 'negatives21', 'mr14', 'twd26', 'twd27', 'mr13',
               'sv04', 'negatives07', 'sv17', 'mr02', 'mr15', 'sv18', 'twd24', 'negatives16', 'twd28', 'got16',
               'negatives01', 'got22', 'negatives13', 'twd25', 'negatives11', 'negatives19', 'mr20', 'sv11', 'mr03',
               'sv02', 'mr08', 'negatives25', 'negatives20', 'twd13', 'got09', 'mr12', 'sv19']  # 114

trainScenceTwoHeads = ['twd14', 'sv08', 'mr09', 'got14', 'negatives04', 'negatives24', 'got03',
                       'sv07', 'sv23', 'got20', 'got13', 'sv21', 'negatives02', 'got04', 'got19', 'negatives05',
                       'negatives09','mr18', 'negatives28', 'twd22', 'got11', 'twd23', 'twd02', 'got15', 'twd08', 'twd01',
                       'negatives26','twd21', 'twd16', 'got17', 'mr21', 'negatives17', 'negatives27', 'twd15', 'got12', 'negatives18',
                       'twd18', 'sv22', 'mr07', 'sv26', 'twd11', 'mr11', 'sv27', 'negatives23', 'twd10', 'twd17',
                       'mr19','sv25', 'negatives14', 'got23', 'got18', 'got21', 'negatives03', 'sv20', 'mr01', 'sv12',
                       'negatives21','mr14', 'twd26', 'twd27', 'mr13', 'sv04', 'negatives07', 'sv17', 'mr02', 'mr15', 'twd24',
                       'negatives16', 'twd28', 'got16', 'negatives01', 'got22', 'negatives13', 'twd25', 'negatives19',
                       'mr20', 'sv11', 'mr03', 'mr08', 'negatives25', 'negatives20', 'twd13', 'got09', 'mr12',
                       'sv19']  # 85

# 6400:7676


testScence = ['got05', 'got06', 'got08', 'mr04', 'mr05', 'mr06', 'mr10', 'sv03', 'sv06', 'sv10', 'twd03', 'twd04',
              'twd05', 'twd06', 'twd07']

testScenceTwoHeads = ['got05', 'got06', 'got08', 'mr04', 'mr05', 'mr06', 'mr10', 'sv03', 'sv06', 'twd03', 'twd04',
                      'twd07']


def Datasetsplit():
    """
    just for two head!
    """
    scence_dict = {}
    scence_dict['val'] = ['negatives27', 'negatives28', 'got21', 'mr19', 'twd08', 'mr05']

    scence_dict['test'] = ['sv06', 'twd21', 'twd11', 'negatives04', 'mr06', 'negatives21',
                           'negatives02', 'negatives19', 'got09', 'sv23', 'got04']

    scence_dict['train'] = ['negatives26', 'got03', 'mr04', 'mr03', 'got22', 'mr11', 'sv03', 'got11', 'twd17',
                            'sv04', 'mr18', 'twd18', 'mr07', 'sv17', 'got13', 'twd03', 'twd01', 'got20', 'twd28',
                            'sv26', 'negatives26', 'negatives18', 'got05', 'sv21', 'mr15', 'got18', 'negatives24',
                            'negatives23', 'twd02', 'got08', 'sv19', 'sv08', 'twd10', 'twd27', 'negatives09', 'twd24',
                            'got12', 'negatives16', 'negatives17', 'mr13', 'sv20', 'sv27', 'mr20', 'negatives01',
                            'got14',
                            'mr09', 'mr02', 'negatives07', 'twd15', 'got17', 'sv07', 'twd14', 'negatives14', 'got19',
                            'sv25', 'got15', 'twd22', 'sv22', 'sv11', 'got23', 'mr01', 'negatives03', 'mr14', 'sv12',
                            'mr08', 'negatives25', 'negatives20', 'twd23', 'mr21', 'twd16', 'twd07', 'got06', 'twd13',
                            'twd04', 'got16', 'negatives13', 'negatives05', 'twd26', 'mr10', 'mr12', 'twd25']
    return scence_dict


def GetAllScences():
    return os.listdir(frame_path)


def convert_xywh2x1y1x2y2(box):
    x, y, w, h = box
    x1 = x
    x2 = x + w - 1
    x1 = max(x1, 0)
    x2 = min(x2, 10000 - 1)
    y1 = max(y, 0)
    y2 = min(y + h - 1, 100000 - 1)
    return [x1, y1, x2, y2]


def xyxy2xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.array([boxes[0], boxes[1], boxes[2] - boxes[0] + 1, boxes[3] - boxes[1] + 1])


def AnnotationGenerator(mode):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    total_id = 0
    print(f'generating annotaion in {mode}')
    total_annotator = []
    # scence_dict = Datasetsplit()

    if mode == 'trainTwoHeads':
        Scence = trainScenceTwoHeads
    elif mode == 'testTwoHeads':
        Scence = testScenceTwoHeads
    elif mode == 'trainScence':
        Scence = trainScence
    elif mode == 'testScence':
        Scence = testScence
    else:
        raise NotImplementedError

    for scence in Scence:
        head_bboxes = np.load(os.path.join(counter_path, scence, 'anno_head.npy'), allow_pickle=True)
        head_bboxes = head_bboxes.item()
        label = None
        if not 'negative' in scence:
            label = np.load(os.path.join(counter_path, scence, 'anno_pair.npy'), allow_pickle=True)
            label = label.item()  # a dict {1:[3, 4],2:[3,4]}
        for frame_id_from1 in head_bboxes.keys():
            """NOTE  -->  in order!!!"""
            """
            person_label
            person_bbox
            laeo 
            height
            width
            """
            # for weird condition
            annotator = {}
            if frame_id_from1 > len(os.listdir(os.path.join(frame_path, scence))) or (
                    label != None and frame_id_from1 > max(label.keys())):
                continue
            img_name = '0' * (6 - len(str(frame_id_from1 - 1))) + str(frame_id_from1 - 1) + '.jpg'
            im = Image.open(os.path.join(frame_path, scence, img_name))
            height = im.height
            width = im.width
            gt_bboxes = []
            for bbox in head_bboxes[frame_id_from1]:
                tmp_dict = dict()
                tmp_dict['tag'] = PERSON
                bbox = xyxy2xywh(bbox)
                tmp_dict['box'] = bbox.tolist()
                gt_bboxes.append(tmp_dict)

            head_num = len(head_bboxes[frame_id_from1])
            laeo_action = []
            if "negative" not in scence:
                for first in range(head_num):
                    for second in range(first + 1, head_num):
                        if ([first + 1, second + 1] == label[frame_id_from1] or [first + 1, second + 1] == label[
                                                                                                               frame_id_from1][
                                                                                                           ::-1]):
                            tmp_dict = dict()
                            tmp_dict['person_1'] = first
                            tmp_dict['person_2'] = second
                            tmp_dict['interaction'] = IS_LAEO
                            laeo_action.append(tmp_dict)
                        else:
                            tmp_dict = dict()
                            tmp_dict['person_1'] = first
                            tmp_dict['person_2'] = second
                            tmp_dict['interaction'] = IS_NOT_LAEO
                            laeo_action.append(tmp_dict)
            else:
                for first in range(head_num):
                    for second in range(first + 1, head_num):
                        tmp_dict = dict()
                        tmp_dict['person_1'] = first
                        tmp_dict['person_2'] = second
                        tmp_dict['interaction'] = IS_NOT_LAEO
                        laeo_action.append(tmp_dict)

            # if total_id not in annotator.keys():
            #     annotator[total_id]={}
            annotator['file_name'] = scence + os.path.sep + img_name
            annotator['width'] = width
            annotator['height'] = height
            annotator['gt_bboxes'] = gt_bboxes
            annotator['laeo'] = laeo_action
            # annotator[total_id]['file_name']=scence+os.path.sep+img_name
            # annotator[total_id]['width']=width
            # annotator[total_id]['height']=height
            # annotator[total_id]['gt_bboxes']=gt_bboxes
            # annotator[total_id]['laeo']=laeo_action
            total_id += 1
            total_annotator.append(annotator)

    with open('/home/edward_krucal/Downloads/Datasets/uco-laeo/tools/' + mode + '.json', 'w') as json_file:
        for each_anno in total_annotator:
            json_file.write(json.dumps(each_anno) + '\n')

    print(f'Done! the number of pictures in {mode} is {total_id + 1}')


if __name__ == '__main__':
    AnnotationGenerator(mode='testScence')

