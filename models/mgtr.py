# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .instance_matcher import build_matcher as build_instance_matcher
from .transformer import build_transformer

class MGTR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_actions, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.person1_cls_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.person1_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.person2_cls_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.person2_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.action_cls_embed = nn.Linear(hidden_dim, num_actions + 1)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        person1_outputs_class = self.person1_cls_embed(hs)
        person1_outputs_coord = self.person1_box_embed(hs).sigmoid()
        person2_outputs_class = self.person2_cls_embed(hs)
        person2_outputs_coord = self.person2_box_embed(hs).sigmoid()
        action_outputs_class = self.action_cls_embed(hs)

        out = {
            'person1_pred_logits': person1_outputs_class[-1],
            'person1_pred_boxes': person1_outputs_coord[-1],
            'person2_pred_logits': person2_outputs_class[-1],
            'person2_pred_boxes': person2_outputs_coord[-1],
            'action_pred_logits': action_outputs_class[-1],
        }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                person1_outputs_class,
                person1_outputs_coord,
                person2_outputs_class,
                person2_outputs_coord,
                action_outputs_class,
            )
        return out

    @torch.jit.unused
    def _set_aux_loss(self,
                      person1_outputs_class,
                      person1_outputs_coord,
                      person2_outputs_class,
                      person2_outputs_coord,
                      action_outputs_class,
                      ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{
            'person1_pred_logits': a,
            'person1_pred_boxes': b,
            'person2_pred_logits': c,
            'person2_pred_boxes': d,
            'action_pred_logits': e,
        } for
            a,
            b,
            c,
            d,
            e,
            in zip(
            person1_outputs_class[:-1],
            person1_outputs_coord[:-1],
            person2_outputs_class[:-1],
            person2_outputs_coord[:-1],
            action_outputs_class[:-1],
        )]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, num_actions, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes  # 2
        self.num_actions = num_actions
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        person1_empty_weight = torch.ones(num_classes + 1)
        person1_empty_weight[-1] = self.eos_coef
        self.register_buffer('person1_empty_weight', person1_empty_weight)

        person2_empty_weight = torch.ones(num_classes + 1)
        person2_empty_weight[-1] = self.eos_coef
        self.register_buffer('person2_empty_weight', person2_empty_weight)

        action_empty_weight = torch.ones(num_actions + 1)
        action_empty_weight[-1] = self.eos_coef
        self.register_buffer('action_empty_weight', action_empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'person1_pred_logits' in outputs
        assert 'person2_pred_logits' in outputs
        assert 'action_pred_logits' in outputs
        person1_src_logits = outputs['person1_pred_logits']
        person2_src_logits = outputs['person2_pred_logits']
        action_src_logits = outputs['action_pred_logits']

        idx = self._get_src_permutation_idx(indices)

        person1_target_classes_o = torch.cat([t["person1_labels"][J] for t, (_, J) in zip(targets, indices)])
        person2_target_classes_o = torch.cat([t["person2_labels"][J] for t, (_, J) in zip(targets, indices)])
        action_target_classes_o = torch.cat([t["action_labels"][J] for t, (_, J) in zip(targets, indices)])

        person1_target_classes = torch.full(person1_src_logits.shape[:2], self.num_classes ,
                                          dtype=torch.int64, device=person1_src_logits.device)
        person1_target_classes[idx] = person1_target_classes_o.long()

        person2_target_classes = torch.full(person2_src_logits.shape[:2], self.num_classes,
                                           dtype=torch.int64, device=person2_src_logits.device)
        person2_target_classes[idx] = person2_target_classes_o.long()

        action_target_classes = torch.full(action_src_logits.shape[:2], self.num_actions,
                                           dtype=torch.int64, device=action_src_logits.device)
        action_target_classes[idx] = action_target_classes_o.long()

        person1_loss_ce = F.cross_entropy(person1_src_logits.transpose(1, 2),
                                        person1_target_classes, self.person1_empty_weight)
        person2_loss_ce = F.cross_entropy(person2_src_logits.transpose(1, 2),
                                         person2_target_classes, self.person2_empty_weight)
        action_loss_ce = F.cross_entropy(action_src_logits.transpose(1, 2),
                                         action_target_classes, self.action_empty_weight)
        loss_ce = person1_loss_ce + person2_loss_ce + 2 * action_loss_ce
        losses = {
            'loss_ce': loss_ce,
            'person1_loss_ce': person1_loss_ce,
            'person2_loss_ce': person2_loss_ce,
            'action_loss_ce': action_loss_ce
        }

        if log:
            losses['class_error'] = 100 - accuracy(action_src_logits[idx], action_target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['action_pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["action_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'person1_pred_boxes' in outputs
        assert 'person2_pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)

        person1_src_boxes = outputs['person1_pred_boxes'][idx]
        person1_target_boxes = torch.cat([t['person1_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        person2_src_boxes = outputs['person2_pred_boxes'][idx]
        person2_target_boxes = torch.cat([t['person2_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        person1_loss_bbox = F.l1_loss(person1_src_boxes, person1_target_boxes, reduction='none')
        person2_loss_bbox = F.l1_loss(person2_src_boxes, person2_target_boxes, reduction='none')

        losses = dict()
        losses['person1_loss_bbox'] = person1_loss_bbox.sum() / num_boxes
        losses['person2_loss_bbox'] = person2_loss_bbox.sum() / num_boxes
        losses['loss_bbox'] = losses['person1_loss_bbox'] + losses['person2_loss_bbox']

        person1_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(person1_src_boxes),
            box_ops.box_cxcywh_to_xyxy(person1_target_boxes)))
        person2_loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(person2_src_boxes),
            box_ops.box_cxcywh_to_xyxy(person2_target_boxes)))
        losses['person1_loss_giou'] = person1_loss_giou.sum() / num_boxes
        losses['person2_loss_giou'] = person2_loss_giou.sum() / num_boxes

        losses['loss_giou'] = losses['person1_loss_giou'] + losses['person2_loss_giou']
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["person1_labels"]) for t in targets)

        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    assert args.dataset_file in ['laeo'], args.dataset_file
    num_classes = 2
    num_actions = 2
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = MGTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_actions=num_actions,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    matcher = build_instance_matcher(args)

    weight_dict = dict(loss_ce=1, loss_bbox=args.bbox_loss_coef, loss_giou=args.giou_loss_coef)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    criterion = SetCriterion(num_classes=num_classes, num_actions=num_actions, matcher=matcher,
                             weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    return model, criterion
