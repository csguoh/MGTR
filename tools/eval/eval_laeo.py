import json
import numpy as np
import logging
import os
from datasets.laeo import convert_xywh2x1y1x2y2


def sparse_gt_per_line(line):
    item = json.loads(line)
    for annot in item['gt_bboxes']:
        box = convert_xywh2x1y1x2y2(annot['box'],[item['height'],item['width']], flip=0)
        annot['box'] = box
    return item


class LAEO_Evaluator:
    def __init__(self, annotation_file):
        self.annotations = [sparse_gt_per_line(l.strip()) for l in open(annotation_file, 'r').readlines()]
        self.overlap_iou = 0.5
        # add explation name for verb_name_dict
        self.verb_name_dict = []
        self.verb_name_dict_name = []

        self.fp = {}
        self.tp = {}
        self.score = {}
        self.sum_gt = {}
        self.file_name = []
        self.train_sum = {}

        for gt_i in self.annotations:
            self.file_name.append(gt_i['file_name'])
            gt_laeo = gt_i['laeo']
            gt_bbox = gt_i['gt_bboxes']
            for gt_laeo_i in gt_laeo:
                if isinstance(gt_laeo_i['interaction'], str):
                    gt_laeo_i['interaction'] = int(gt_laeo_i['interaction'].replace('\n', ''))
                triplet = [gt_bbox[gt_laeo_i['person_1']]['tag'],
                           gt_bbox[gt_laeo_i['person_2']]['tag'], gt_laeo_i['interaction']]
                if triplet not in self.verb_name_dict:
                    self.verb_name_dict.append(triplet)
                if self.verb_name_dict.index(triplet) not in self.sum_gt.keys():
                    self.sum_gt[self.verb_name_dict.index(triplet)] = 0
                self.sum_gt[self.verb_name_dict.index(triplet)] += 1

        for i in range(len(self.verb_name_dict)): # two class:  laeo  &  w/t_laeo
            self.fp[i] = []
            self.tp[i] = []
            self.score[i] = []
        self.num_class = len(self.verb_name_dict)


    def evalution(self, predict_annot):
        for pred_i in predict_annot:
            if pred_i['file_name'] not in self.file_name:
                continue
            gt_i = self.annotations[self.file_name.index(pred_i['file_name'])]

            assert gt_i['file_name'] == pred_i['file_name']

            gt_bbox = gt_i['gt_bboxes']
            if len(gt_bbox) != 0:
                pred_bbox = self.add_One(pred_i['predictions'])  # convert zero-based to one-based indices
                if len(pred_bbox) == 0:  # To prevent compute_iou_mat
                    logging.warning(f"Image {pred_i['file_name']} pred NULL")
                    continue
                bbox_pairs, bbox_ov = self.compute_iou_mat(gt_bbox, pred_bbox)
                pred_laeo = pred_i['laeo_prediction']
                gt_laeo = gt_i['laeo']
                self.compute_fptp(pred_laeo, gt_laeo, bbox_pairs, pred_bbox, bbox_ov)
            else: # no person in one image
                pred_bbox = self.add_One(pred_i['predictions'])  # convert zero-based to one-based indices
                for i, pred_laeo_i in enumerate(pred_i['laeo_prediction']):
                    triplet = [pred_bbox[pred_laeo_i['person1_id']]['tag'],
                               pred_bbox[pred_laeo_i['person2_id']]['tag'], pred_laeo_i['interaction']]
                    verb_id = self.verb_name_dict.index(triplet)
                    self.tp[verb_id].append(0)
                    self.fp[verb_id].append(1)
                    self.score[verb_id].append(pred_laeo_i['score'])
        # The tp and fp of all pictures in the entire test set are calculated, and the overall ap is calculated.
        ap_score = self.compute_ap()
        return ap_score


    def compute_ap(self, save_mAP=None):
        ap = np.zeros(self.num_class)
        max_recall = np.zeros(self.num_class)
        name2ap = {}
        for i in range(len(self.verb_name_dict)):
            name = 'laeo' if i == 1 else 'w/t laeo'
            sum_gt = self.sum_gt[i]
            if sum_gt == 0:
                continue
            tp = np.asarray((self.tp[i]).copy())
            fp = np.asarray((self.fp[i]).copy())
            res_num = len(tp)
            if res_num == 0:
                continue
            score = np.asarray(self.score[i].copy())
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            recall = tp / sum_gt
            precision = tp / (fp + tp)
            ap[i] = self.voc_ap(recall, precision)
            max_recall[i] = np.max(recall)
            name2ap[name] = ap[i]

        AP = np.mean(ap[:])
        m_rec = np.mean(max_recall[:])
        print('--------------------')
        print(f'mAP: {AP}, AP_0: {ap[0]}, AP_1: {ap[1]}, max_recall: {m_rec}')
        print('--------------------')

        if save_mAP is not None:
            json.dump(name2ap, open(save_mAP, "w"))
        return AP

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_laeo, gt_laeo, match_pairs, pred_bbox, bbox_ov):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_laeo))
        pred_laeo.sort(key=lambda k: (k.get('score', 0)), reverse=True) # nms
        if len(pred_laeo) != 0:
            for i, pred_laeo_i in enumerate(pred_laeo):
                is_match = 0
                if isinstance(pred_laeo_i['interaction'], str):
                    pred_laeo_i['interaction'] = int(pred_laeo_i['interaction'].replace('\n', ''))
                if len(match_pairs) != 0 and pred_laeo_i['person1_id'] in pos_pred_ids and pred_laeo_i['person2_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_laeo_i['person1_id']] # 预测的person1 对应于gt的第几个
                    pred_obj_ids = match_pairs[pred_laeo_i['person2_id']]
                    pred_obj_ov = bbox_ov[pred_laeo_i['person2_id']]
                    pred_sub_ov = bbox_ov[pred_laeo_i['person1_id']]
                    pred_tag = pred_laeo_i['interaction']
                    max_ov = 0
                    max_gt_id = 0
                    for gt_id in range(len(gt_laeo)):
                        gt_laeo_i = gt_laeo[gt_id]
                        if (gt_laeo_i['person_1'] in pred_sub_ids) and (gt_laeo_i['person_2'] in pred_obj_ids) and (pred_tag == gt_laeo_i['interaction']):
                            is_match = 1
                            min_ov_gt = min(pred_sub_ov[pred_sub_ids.index(gt_laeo_i['person_1'])],
                                            pred_obj_ov[pred_obj_ids.index(gt_laeo_i['person_2'])])
                            if min_ov_gt > max_ov:
                                max_ov = min_ov_gt
                                max_gt_id = gt_id
                        elif (gt_laeo_i['person_2'] in pred_sub_ids) and (gt_laeo_i['person_1'] in pred_obj_ids) and (pred_tag == gt_laeo_i['interaction']):
                            is_match = 1
                            min_ov_gt = min(pred_sub_ov[pred_sub_ids.index(gt_laeo_i['person_2'])],
                                            pred_obj_ov[pred_obj_ids.index(gt_laeo_i['person_1'])])
                            if min_ov_gt > max_ov:
                                max_ov = min_ov_gt
                                max_gt_id = gt_id
                if pred_laeo_i['interaction'] not in list(self.fp.keys()):
                    continue
                triplet = [pred_bbox[pred_laeo_i['person1_id']]['tag'],
                           pred_bbox[pred_laeo_i['person2_id']]['tag'], pred_laeo_i['interaction']]
                if triplet not in self.verb_name_dict:
                    continue
                verb_id = self.verb_name_dict.index(triplet)
                if is_match == 1 and vis_tag[max_gt_id] == 0:
                    self.fp[verb_id].append(0)
                    self.tp[verb_id].append(1)
                    vis_tag[max_gt_id] = 1
                else:
                    self.fp[verb_id].append(1)
                    self.tp[verb_id].append(0)
                self.score[verb_id].append(pred_laeo_i['score'])


    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i

        iou_mat_ov = iou_mat.copy()
        iou_mat[iou_mat >= 0.5] = 1
        iou_mat[iou_mat < 0.5] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pairs_ov = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pairs_ov[pred_id] = []
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pairs_ov[pred_id].append(iou_mat_ov[match_pairs[0][i], pred_id])
        return match_pairs_dict, match_pairs_ov


    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['tag'], str):
            bbox1['tag'] = int(bbox1['tag'].replace('\n', ''))
        if isinstance(bbox2['tag'], str):
            bbox2['tag'] = int(bbox2['tag'].replace('\n', ''))
        if bbox1['tag'] == bbox2['tag']:
            rec1 = bbox1['box']
            rec2 = bbox2['box']
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0] + 1) * (rec1[3] - rec1[1] + 1)
            S_rec2 = (rec2[2] - rec2[0] + 1) * (rec2[3] - rec2[1] + 1)

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line + 1) * (bottom_line - top_line + 1)
                return intersect / (sum_area - intersect)
        else:
            return 0

    def add_One(self, prediction):  # Add 1 to all coordinates
        for i, pred_bbox in enumerate(prediction):
            rec = pred_bbox['box']
            rec[0] += 1
            rec[1] += 1
            rec[2] += 1
            rec[3] += 1
        return prediction

laeo_action_name = ['laeo', 'w/t laeo']


def get_laeo_output(Image_dets):
    output_laeo = []
    for Image_det in Image_dets:
        Image_det = json.loads(Image_det)
        file_name = Image_det['image_id']
        output = {'predictions': [], 'laeo_prediction': [], 'file_name': file_name}
        count = 0
        for det in Image_det['laeo_list']:
            person1_bbox = det['person1_box']
            person1_score = det['person1_cls']

            person2_bbox = det['person2_box']
            person2_score = det['person2_cls']

            inter_name = det["laeo_name"]
            inter_score = det["laeo_cls"]
            inter_cat = 1 if inter_name == 'laeo' else 0

            output['predictions'].append({'box': person1_bbox, 'tag': 1})
            person1_idx = count
            count += 1
            output['predictions'].append({'box': person2_bbox, 'tag': 1}) # person2:1
            person2_idx = count
            count += 1

            final_score = person1_score * person2_score * inter_score

            output['laeo_prediction'].append({'person1_id': person1_idx, 'person2_id': person2_idx,
                                              'interaction': inter_cat, 'score': final_score})
        output_laeo.append(output)
    return output_laeo


def LAEO_Evaluate(output_file,args):
    # 1. transform model output
    with open(output_file, "r") as f:
        det = f.readlines()
    output_laeo = get_laeo_output(det) # convert format from 'str' to 'dict'

    # 2. evaluation
    if args.data_name == 'uco':
        laeo_evaluator = LAEO_Evaluator("./data/uco_testScence.json")
    elif args.data_name == 'ava':
        laeo_evaluator = LAEO_Evaluator("./data/ava_testScence.json")
    ap = laeo_evaluator.evalution(output_laeo)
    return ap


if __name__ == '__main__':
    laeo_evaluator = LAEO_Evaluator("./data/ava_testScence.json")