import argparse
import json
import random
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.laeo import build as build_dataset
from models.mgtr import build as build_model
import util.misc as utils
from tools.eval.eval_laeo import LAEO_Evaluate

def nms(dets, thresh):
    if 0==len(dets): return []
    x1,y1,x2,y2,scores = dets[:, 0],dets[:, 1],dets[:, 2],dets[:, 3],dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1,yy1 = np.maximum(x1[i], x1[order[1:]]),np.maximum(y1[i], y1[order[1:]])
        xx2,yy2 = np.minimum(x2[i], x2[order[1:]]),np.minimum(y2[i], y2[order[1:]])

        w,h = np.maximum(0.0, xx2 - xx1 + 1),np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w*h / (areas[i] + areas[order[1:]] - w*h)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def get_args_parser_test():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Backbone.
    parser.add_argument('--backbone', default='resnet50', choices=['resnet50', 'resnet101'],
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer.
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss.
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # Matcher.
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # Loss coefficients.
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.02, type=float,
                        help="Relative classification weight of the no-object class")

    # Dataset parameters.
    parser.add_argument('--dataset_file', default='laeo')
    parser.add_argument('--data_name',default='uco',choices=['ava','uco'],
                        help='select which dataset will be used --> UCO-LAEO or AVA-LAEO')
    parser.add_argument('--model_path', default='E:\\MGTR-master\\data\\mgtr_pretrained\\best_ava0.6431.pth',
                        help='Path of the models to evaluate.')
    parser.add_argument('--log_dir', default='./',
                        help='path where to save temporary files in test')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)

    # Visualization.
    parser.add_argument('--max_to_viz', default=50, type=int, help='number of images to visualize')
    parser.add_argument('--save_image',default=True, action='store_true', help='whether to save visualization images')
    parser.add_argument("--eval_path", default="./data") # the dir of -> ./data/laeo_test.json
    return parser


def random_color():
    rdn = random.randint(1, 1000)
    b = int(rdn * 997) % 255
    g = int(rdn * 4447) % 255
    r = int(rdn * 6563) % 255
    return b, g, r


def intersection(box_a, box_b):
    # box: x1, y1, x2, y2
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    return float((x2 - x1 + 1) * (y2 - y1 + 1))


def IoU(box_a, box_b):
    inter = intersection(box_a, box_b)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    union = box_a_area + box_b_area - inter
    return inter / float(max(union, 1))


def triplet_nms(laeo_list):
    laeo_list.sort(key=lambda x: x['person1_cls'] * x['person2_cls'] * x['laeo_cls'], reverse=True)
    mask = [True] * len(laeo_list)
    for idx_x in range(len(laeo_list)):
        if mask[idx_x] is False:
            continue
        for idx_y in range(idx_x + 1, len(laeo_list)):
            x = laeo_list[idx_x]
            y = laeo_list[idx_y]
            iou_1 = IoU(x['person1_box'], y['person1_box'])
            iou_2 = IoU(x['person2_box'], y['person2_box'])
            iou_3 = IoU(x['person1_box'],y['person2_box'])
            iou_4 = IoU(x['person2_box'],y['person1_box'])
            if iou_1>0.5 and iou_2>0.5 or iou_3>0.5 and iou_4>0.5:
                mask[idx_y] = False
    new_laeo_list = []
    for idx in range(len(mask)):
        if mask[idx] is True:
            new_laeo_list.append(laeo_list[idx])
    return new_laeo_list



def inference_on_data(args, model_path, image_set, max_to_viz=10, test_scale=-1):
    assert image_set in ['train', 'test'], image_set
    checkpoint = torch.load(model_path, map_location='cpu')
    epoch = checkpoint['epoch']
    print('epoch:', epoch)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.device)
    model, criterion = build_model(args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    dataset_val = build_dataset(image_set=image_set, args=args, test_scale=test_scale)
    sampler_val = torch.utils.data.RandomSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    log_dir = os.path.join(args.log_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    assert os.path.exists(log_dir), log_dir
    file_name = 'result_s%03d_e%03d_%s_%s.pkl' \
                % (0 if test_scale == -1 else test_scale, epoch, args.dataset_file, args.backbone)
    file_path = os.path.join(log_dir, file_name)

    result_list = []
    for samples, targets in data_loader_val:
        id_list = [targets[idx]['image_id'] for idx in range(len(targets))]
        org_sizes = [targets[idx]['org_size'] for idx in range(len(targets))]
        samples = samples.to(device)
        outputs = model(samples)
        action_pred_logits = outputs['action_pred_logits']
        person2_pred_logits = outputs['person2_pred_logits']
        person2_pred_boxes = outputs['person2_pred_boxes']
        person1_pred_logits = outputs['person1_pred_logits']
        person1_pred_boxes = outputs['person1_pred_boxes']
        result_list.append(dict(
            id_list=id_list,
            org_sizes=org_sizes,
            action_pred_logits=action_pred_logits.detach().cpu(),
            person2_pred_logits=person2_pred_logits.detach().cpu(),
            person2_pred_boxes=person2_pred_boxes.detach().cpu(),
            person1_pred_logits=person1_pred_logits.detach().cpu(),
            person1_pred_boxes=person1_pred_boxes.detach().cpu(),
        ))

    with open(file_path, 'wb') as f:
        torch.save(result_list, f)
    print('step1: inference done.')
    return file_path


def parse_model_result(args, result_path, laeo_th=0.6, person1_th=0.6, person2_th=0.6, max_to_viz=10):
    assert args.dataset_file in ['laeo'], args.dataset_file
    num_classes = 2
    num_actions = 2
    top_k = 35

    with open(result_path, 'rb') as f:
        output_list = torch.load(f, map_location='cpu')

    final_laeo_result_list = []
    for outputs in output_list:  # batch level
        img_id_list = outputs['id_list']
        org_sizes = outputs['org_sizes']
        action_pred_logits = outputs['action_pred_logits']
        person2_pred_logits = outputs['person2_pred_logits']
        person2_pred_boxes = outputs['person2_pred_boxes']
        person1_pred_logits = outputs['person1_pred_logits']
        person1_pred_boxes = outputs['person1_pred_boxes']
        assert len(action_pred_logits) == len(img_id_list)

        for idx_img in range(len(action_pred_logits)):
            image_id = img_id_list[idx_img]
            hh, ww = org_sizes[idx_img]

            act_cls = torch.nn.Softmax(dim=1)(action_pred_logits[idx_img]).detach().cpu().numpy()
            person1_cls = torch.nn.Softmax(dim=1)(person1_pred_logits[idx_img]).detach().cpu().numpy()
            person2_cls = torch.nn.Softmax(dim=1)(person2_pred_logits[idx_img]).detach().cpu().numpy()
            person1_box = person1_pred_boxes[idx_img].detach().cpu().numpy()
            person2_box = person2_pred_boxes[idx_img].detach().cpu().numpy()

            keep = (act_cls.argmax(axis=1) != num_actions)
            keep = keep * (person1_cls.argmax(axis=1) != num_classes)
            keep = keep * (person2_cls.argmax(axis=1) != num_classes)
            keep = keep * (act_cls > laeo_th).any(axis=1)
            keep = keep * (person1_cls > person1_th).any(axis=1)
            keep = keep * (person2_cls > person2_th).any(axis=1)

            person1_val_max_list = person1_cls[keep][:, :-1].max(axis=1)
            person1_box_max_list = person1_box[keep]
            person2_val_max_list = person2_cls[keep][:, :-1].max(axis=1)
            person2_box_max_list = person2_box[keep]
            keep_act_scores = act_cls[keep][:, :-1]

            keep_act_scores_1d = keep_act_scores.reshape(-1)
            top_k_idx_1d = np.argsort(-keep_act_scores_1d)[:top_k]
            box_action_pairs = [(idx_1d // num_actions, idx_1d % num_actions) for idx_1d in top_k_idx_1d]

            laeo_list = []
            for idx_box, idx_action in box_action_pairs:
                # action
                laeo_cls = keep_act_scores[idx_box, idx_action]
                laeo_name = 'laeo' if idx_action == 1 else 'w/t_laeo'
                # person1
                cx, cy, w, h = person1_box_max_list[idx_box]
                cx, cy, w, h = cx * ww, cy * hh, w * ww, h * hh
                person1_box = list(map(int, [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]))
                person1_cls = person1_val_max_list[idx_box]
                # person2
                cx, cy, w, h = person2_box_max_list[idx_box]
                cx, cy, w, h = cx * ww, cy * hh, w * ww, h * hh
                person2_box = list(map(int, [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]))
                person2_cls = person2_val_max_list[idx_box]
                if laeo_cls < laeo_th or person1_cls < person1_th or person2_cls < person2_th:
                    continue
                pp = dict(
                    person1_cls=float(person1_cls), person1_name='person',
                    person2_cls=float(person2_cls), person2_name='person', laeo_cls=float(laeo_cls),
                    person1_box=person1_box, person2_box=person2_box, laeo_name=laeo_name,
                )
                laeo_list.append(pp)

            laeo_list = triplet_nms(laeo_list) # nms
            item = dict(image_id=image_id, laeo_list=laeo_list)
            final_laeo_result_list.append(item)
    return final_laeo_result_list # 整个测试集的结果


def draw_on_image(args, image_id, laeo_list, image_path):
    img_name = image_id
    assert args.dataset_file in ['laeo'], args.dataset_file
    if args.dataset_file == 'laeo':
        img_path = 'E:/uco-laeo/ucolaeodb/frames/' + img_name
    else:
        raise NotImplementedError()

    img_result = cv2.imread(img_path, cv2.IMREAD_COLOR)
    bbox_list = []
    color = random_color()
    for idx_box, laeo in enumerate(laeo_list):
        # action
        laeo_cls, laeo_name = laeo['laeo_cls'], laeo['laeo_name']
        cv2.putText(img_result, '%s:%.4f' % (laeo_name, laeo_cls),
                    (10, 50 * idx_box + 50), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        # person1
        bbox_list.append(np.array(laeo['person1_box']+[laeo['person1_cls']]))
        # person2
        bbox_list.append(np.array(laeo['person2_box']+[laeo['person2_cls']]))
        # laeo
        if laeo_name == 'laeo':
            person1_mid = (laeo['person1_box'][0]+laeo['person1_box'][2])//2,(laeo['person1_box'][1]+laeo['person1_box'][3])//2
            person2_mid =(laeo['person2_box'][0]+laeo['person2_box'][2])//2,(laeo['person2_box'][1]+laeo['person2_box'][3])//2
            cv2.line(img_result,person1_mid,person2_mid,(0,255,0),2)
    if 0 == len(bbox_list):
        bbox_list = np.zeros((1, 5))
    bbox_list = np.vstack(bbox_list)
    keep = nms(bbox_list, 0.3)
    bbox_list = bbox_list[keep, :]
    for box in bbox_list:
        box = list(map(int,box))
        x1, y1, x2, y2,score= box
        cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)

    if img_result.shape[0] > 640:
        ratio = img_result.shape[0] / 640
        img_result = cv2.resize(img_result, (int(img_result.shape[1] / ratio), int(img_result.shape[0] / ratio)))
    cv2.imwrite(image_path, img_result)


def eval_once(args, model_result_path, laeo_th=0.9, person1_th=0.8, person2_th=0.8, max_to_viz=10, save_image=True):
    assert args.dataset_file in ['laeo'], args.dataset_file
    # post-process the model result
    # 1. Only take the non-background
    # 2. nms, delete the repeated boxes
    laeo_result_list = parse_model_result(
        args=args,
        result_path=model_result_path,
        laeo_th=laeo_th,
        person1_th=person1_th,
        person2_th=person2_th,
        max_to_viz=max_to_viz,
    )

    result_file = model_result_path.replace('.pkl', '.json')
    with open(result_file, 'w') as writer:
        for idx_img, item in enumerate(laeo_result_list):
            writer.write(json.dumps(item) + '\n')
            if save_image and idx_img < max_to_viz:
                img_path = '%s/dt_%02d.jpg' % (os.path.dirname(model_result_path), idx_img)
                draw_on_image(args, item['image_id'], item['laeo_list'], image_path=img_path)

    LAEO_Evaluate(result_file,args)


def run_and_eval(args, model_path, test_scale, max_to_viz=10, save_image=False):
    model_output_file = inference_on_data(
        args=args,
        model_path=model_path,
        image_set='test',
        test_scale=test_scale,
        max_to_viz=max_to_viz,
    ) # Run the entire test data set, save it, the metric score is based on the entire dataset


    eval_once(args=args,
              model_result_path=model_output_file,
              laeo_th=0,
              person1_th=0,
              person2_th=0,
              max_to_viz=max_to_viz,
              save_image=save_image
              )
    pass


def test():
    parser_test = get_args_parser_test()
    args_test = parser_test.parse_args()
    print(args_test)
    test_scale = 672
    model_path = args_test.model_path
    run_and_eval(args=args_test, model_path=model_path,
                 test_scale=test_scale,
                 max_to_viz=args_test.max_to_viz if args_test.save_image else 200 * 100,
                 save_image=args_test.save_image,
                 )
    print('test done')


if __name__ == '__main__':
    test()
