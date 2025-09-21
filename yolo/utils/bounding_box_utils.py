import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange
from torch import Tensor, tensor
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import batched_nms

from yolo.config.config import AnchorConfig, MatcherConfig, NMSConfig
from yolo.model.yolo import YOLO
from yolo.utils.logger import logger


def calculate_iou(bbox1, bbox2, metrics="iou") -> Tensor:
    metrics = metrics.lower()
    EPS = 1e-7
    dtype = bbox1.dtype
    bbox1 = bbox1.to(torch.float32)
    bbox2 = bbox2.to(torch.float32)

    # Expand dimensions if necessary
    if bbox1.ndim == 2 and bbox2.ndim == 2:
        bbox1 = bbox1.unsqueeze(1)  # (Ax4) -> (Ax1x4)
        bbox2 = bbox2.unsqueeze(0)  # (Bx4) -> (1xBx4)
    elif bbox1.ndim == 3 and bbox2.ndim == 3:
        bbox1 = bbox1.unsqueeze(2)  # (BZxAx4) -> (BZxAx1x4)
        bbox2 = bbox2.unsqueeze(1)  # (BZxBx4) -> (BZx1xBx4)

    # Calculate intersection coordinates
    xmin_inter = torch.max(bbox1[..., 0], bbox2[..., 0])
    ymin_inter = torch.max(bbox1[..., 1], bbox2[..., 1])
    xmax_inter = torch.min(bbox1[..., 2], bbox2[..., 2])
    ymax_inter = torch.min(bbox1[..., 3], bbox2[..., 3])

    # Calculate intersection area
    intersection_area = torch.clamp(xmax_inter - xmin_inter, min=0) * torch.clamp(ymax_inter - ymin_inter, min=0)

    # Calculate area of each bbox
    area_bbox1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area_bbox2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    # Calculate union area
    union_area = area_bbox1 + area_bbox2 - intersection_area

    # Calculate IoU
    iou = intersection_area / (union_area + EPS)
    if metrics == "iou":
        return iou.to(dtype)

    # Calculate centroid distance
    cx1 = (bbox1[..., 2] + bbox1[..., 0]) / 2
    cy1 = (bbox1[..., 3] + bbox1[..., 1]) / 2
    cx2 = (bbox2[..., 2] + bbox2[..., 0]) / 2
    cy2 = (bbox2[..., 3] + bbox2[..., 1]) / 2
    cent_dis = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Calculate diagonal length of the smallest enclosing box
    c_x = torch.max(bbox1[..., 2], bbox2[..., 2]) - torch.min(bbox1[..., 0], bbox2[..., 0])
    c_y = torch.max(bbox1[..., 3], bbox2[..., 3]) - torch.min(bbox1[..., 1], bbox2[..., 1])
    diag_dis = c_x**2 + c_y**2 + EPS

    diou = iou - (cent_dis / diag_dis)
    if metrics == "diou":
        return diou.to(dtype)

    # Compute aspect ratio penalty term
    arctan = torch.atan((bbox1[..., 2] - bbox1[..., 0]) / (bbox1[..., 3] - bbox1[..., 1] + EPS)) - torch.atan(
        (bbox2[..., 2] - bbox2[..., 0]) / (bbox2[..., 3] - bbox2[..., 1] + EPS)
    )
    v = (4 / (math.pi**2)) * (arctan**2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + EPS)
    # Compute CIoU
    ciou = diou - alpha * v
    return ciou.to(dtype)


def transform_bbox(bbox: Tensor, indicator="xywh -> xyxy"):
    data_type = bbox.dtype
    in_type, out_type = indicator.replace(" ", "").split("->")

    if in_type not in ["xyxy", "xywh", "xycwh"] or out_type not in ["xyxy", "xywh", "xycwh"]:
        raise ValueError("Invalid input or output format")

    if in_type == "xywh":
        x_min = bbox[..., 0]
        y_min = bbox[..., 1]
        x_max = bbox[..., 0] + bbox[..., 2]
        y_max = bbox[..., 1] + bbox[..., 3]
    elif in_type == "xyxy":
        x_min = bbox[..., 0]
        y_min = bbox[..., 1]
        x_max = bbox[..., 2]
        y_max = bbox[..., 3]
    elif in_type == "xycwh":
        x_min = bbox[..., 0] - bbox[..., 2] / 2
        y_min = bbox[..., 1] - bbox[..., 3] / 2
        x_max = bbox[..., 0] + bbox[..., 2] / 2
        y_max = bbox[..., 1] + bbox[..., 3] / 2

    if out_type == "xywh":
        bbox = torch.stack([x_min, y_min, x_max - x_min, y_max - y_min], dim=-1)
    elif out_type == "xyxy":
        bbox = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    elif out_type == "xycwh":
        bbox = torch.stack([(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min], dim=-1)

    return bbox.to(dtype=data_type)


def generate_anchors(image_size: List[int], strides: List[int]):
    """
    Find the anchor maps for each w, h.

    Args:
        image_size List: the image size of augmented image size
        strides List[8, 16, 32, ...]: the stride size for each predicted layer

    Returns:
        all_anchors [HW x 2]:
        all_scalers [HW]: The index of the best targets for each anchors
    """
    W, H = image_size
    anchors = []
    scaler = []
    for stride in strides:
        anchor_num = W // stride * H // stride
        scaler.append(torch.full((anchor_num,), stride))
        shift = stride // 2
        h = torch.arange(0, H, stride) + shift
        w = torch.arange(0, W, stride) + shift
        if torch.__version__ >= "2.3.0":
            anchor_h, anchor_w = torch.meshgrid(h, w, indexing="ij")
        else:
            anchor_h, anchor_w = torch.meshgrid(h, w)
        anchor = torch.stack([anchor_w.flatten(), anchor_h.flatten()], dim=-1)
        anchors.append(anchor)
    all_anchors = torch.cat(anchors, dim=0)
    all_scalers = torch.cat(scaler, dim=0)
    return all_anchors, all_scalers


class BoxMatcher:
    def __init__(self, cfg: MatcherConfig, class_num: int, vec2box, reg_max: int) -> None:
        self.class_num = class_num
        self.vec2box = vec2box
        self.reg_max = reg_max
        for attr_name in cfg:
            setattr(self, attr_name, cfg[attr_name])

    def get_valid_matrix(self, target_bbox: torch.Tensor) -> torch.Tensor:
        """
        Return a boolean mask [B, T, A] telling whether each GT (T) can be validly predicted
        by each anchor (A), considering:
        1) anchor center inside GT box (in-box),
        2) anchor center inside a center region around GT center (center prior),
        3) all four distances (l,t,r,b) to GT edges are within DFL range (reg_max).

        Args:
            target_bbox: [B, T, 4] in xyxy (image pixel coordinates).

        Returns:
            valid: [B, T, A] boolean
        """
        device = target_bbox.device
        dtype = target_bbox.dtype
        B, T = target_bbox.shape[:2]
        if T == 0:
            # no targets
            A = self.vec2box.anchor_grid.shape[0]
            return torch.zeros((B, 0, A), dtype=torch.bool, device=device)

        # --- anchors & stride ---
        # anchor_grid: (A, 2) -> (1,1,A,2)
        anchors = self.vec2box.anchor_grid.to(device).view(1, 1, -1, 2)
        ax = anchors[..., 0]  # (B=1, T=1, A)
        ay = anchors[..., 1]

        # stride per anchor: (A,) -> (1,1,A)
        stride = self.vec2box.scaler.to(device).view(1, 1, -1)

        # --- GT boxes expanded ---
        # target_bbox: [B,T,4] -> [B,T,1,4]
        tb = target_bbox.unsqueeze(2)  # (B, T, 1, 4)
        x_min, y_min, x_max, y_max = tb.unbind(dim=-1)  # each (B, T, 1)

        # --- DFL range check (reg_max) ---
        # Distances (in pixels) from anchor center to each GT edge, then normalize by stride.
        # l = (ax - x_min)/s, t = (ay - y_min)/s, r = (x_max - ax)/s, b = (y_max - ay)/s
        # NOTE: broadcasting: ax, ay are (1,1,A) and x_min,... are (B,T,1)
        l = (ax - x_min) / stride
        t = (ay - y_min) / stride
        r = (x_max - ax) / stride
        b = (y_max - ay) / stride

        # anchor center inside GT box
        on_anchor = (l >= 0) & (t >= 0) & (r >= 0) & (b >= 0)

        # All four distances must be <= (reg_max - 1 - tiny_slack)
        # Allow a small slack at the DFL upper bound to avoid rejecting near-miss anchors
        reg_slack = float(getattr(self, "reg_slack", 0.01))
        reg_thr = float(self.reg_max) - 1.0 - reg_slack  # e.g., 16 -> 14.99
        # max over (l,t,r,b)
        max_reg = torch.stack((l, t, r, b), dim=-1).amax(dim=-1)
        in_reg = max_reg <= reg_thr  # (B, T, A)

        # --- center prior (stride-based rectangle around GT center) ---
        # Typical TAL uses a radius in *stride* units per level.
        center_radius = float(getattr(self, "center_radius", 2.5))
        rad = center_radius * stride  # (B=1, T=1, A)

        cx = (x_min + x_max) * 0.5  # (B, T, 1)
        cy = (y_min + y_max) * 0.5
        in_center = (ax >= (cx - rad)) & (ax <= (cx + rad)) & \
                    (ay >= (cy - rad)) & (ay <= (cy + rad))  # (B, T, A)

        # --- candidate policy: how to combine in-box and center prior ---
        policy = getattr(self, "candidate_policy", "or").lower()
        if policy == "and":
            candidates = on_anchor & in_center
        elif policy == "box":
            candidates = on_anchor
        elif policy == "center":
            candidates = in_center
        else:  # default to 'or'
            candidates = on_anchor | in_center

        # --- final valid mask: candidate AND in_reg (within DFL range) ---
        valid = candidates & in_reg  # (B, T, A)

        # ensure boolean dtype
        return valid.bool()

    def get_cls_matrix(self, predict_cls: Tensor, target_cls: Tensor) -> Tensor:
        """
        Get the (predicted class' probabilities) corresponding to the target classes across all anchors

        Args:
            predict_cls [batch x anchors x class]: The predicted probabilities for each class across each anchor.
            target_cls [batch x targets]: The class index for each target.

        Returns:
            [batch x targets x anchors]: The probabilities from `pred_cls` corresponding to the class indices specified in `target_cls`.
        """
        predict_cls = predict_cls.transpose(1, 2)
        target_cls = target_cls.expand(-1, -1, predict_cls.size(2))
        cls_probabilities = torch.gather(predict_cls, 1, target_cls)
        return cls_probabilities

    def get_iou_matrix(self, predict_bbox, target_bbox) -> Tensor:
        """
        Get the IoU between each target bounding box and each predicted bounding box.

        Args:
            predict_bbox [batch x predicts x 4]: Bounding box with [x1, y1, x2, y2].
            target_bbox [batch x targets x 4]: Bounding box with [x1, y1, x2, y2].
        Returns:
            [batch x targets x predicts]: The IoU scores between each target and predicted.
        """
        return calculate_iou(target_bbox, predict_bbox, metrics="iou").clamp(0, 1)

    def filter_topk(self, target_matrix: Tensor, grid_mask: Tensor, topk: int = 10) -> Tuple[Tensor, Tensor]:
        """
        Filter the top-k suitability of targets for each anchor.

        Args:
            target_matrix [batch x targets x anchors]: The suitability for each targets-anchors
            grid_mask [batch x targets x anchors]: The match validity for each target to anchors
            topk (int, optional): Number of top scores to retain per anchor.

        Returns:
            topk_targets [batch x targets x anchors]: Only leave the topk targets for each anchor
            topk_mask [batch x targets x anchors]: A boolean mask indicating the top-k scores' positions.
        """
        masked_target_matrix = grid_mask * target_matrix
        values, indices = masked_target_matrix.topk(topk, dim=-1)
        topk_targets = torch.zeros_like(target_matrix, device=target_matrix.device)
        topk_targets.scatter_(dim=-1, index=indices, src=values)
        topk_mask = topk_targets > 0
        return topk_targets, topk_mask

    def ensure_one_anchor(self, target_matrix: Tensor, topk_mask: tensor) -> Tensor:
        """
        Ensures each valid target gets at least one anchor matched based on the unmasked target matrix,
        which enables an otherwise invalid match. This enables too small or too large targets to be
        learned as well, even if they can't be predicted perfectly.

        Args:
            target_matrix [batch x targets x anchors]: The suitability for each targets-anchors
            topk_mask [batch x targets x anchors]: A boolean mask indicating the top-k scores' positions.

        Returns:
            topk_mask [batch x targets x anchors]: A boolean mask indicating the updated top-k scores' positions.
        """
        values, indices = target_matrix.max(dim=-1)
        best_anchor_mask = torch.zeros_like(target_matrix, dtype=torch.bool)
        best_anchor_mask.scatter_(-1, index=indices[..., None], src=~best_anchor_mask)
        matched_anchor_num = torch.sum(topk_mask, dim=-1)
        target_without_anchor = (matched_anchor_num == 0) & (values > 0)
        topk_mask = torch.where(target_without_anchor[..., None], best_anchor_mask, topk_mask)
        return topk_mask

    def filter_duplicates(self, iou_mat: Tensor, topk_mask: Tensor):
        """
        Filter the maximum suitability target index of each anchor based on IoU.

        Args:
            iou_mat [batch x targets x anchors]: The IoU for each targets-anchors
            topk_mask [batch x targets x anchors]: A boolean mask indicating the top-k scores' positions.

        Returns:
            unique_indices [batch x anchors x 1]: The index of the best targets for each anchors
            valid_mask [batch x anchors]: Mask indicating the validity of each anchor
            topk_mask [batch x targets x anchors]: A boolean mask indicating the updated top-k scores' positions.
        """
        duplicates = (topk_mask.sum(1, keepdim=True) > 1).repeat([1, topk_mask.size(1), 1])
        masked_iou_mat = topk_mask * iou_mat
        best_indices = masked_iou_mat.argmax(1)[:, None, :]
        best_target_mask = torch.zeros_like(duplicates, dtype=torch.bool)
        best_target_mask.scatter_(1, index=best_indices, src=~best_target_mask)
        topk_mask = torch.where(duplicates, best_target_mask, topk_mask)
        unique_indices = topk_mask.to(torch.uint8).argmax(dim=1)
        return unique_indices[..., None], topk_mask.any(dim=1), topk_mask

    def __call__(self, target: Tensor, predict: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Matches each target to the most suitable anchor.
        1. For each anchor prediction, find the highest suitability targets.
        2. Match target to the best anchor.
        3. Noramlize the class probilities of targets.

        Args:
            target: The ground truth class and bounding box information
                as tensor of size [batch x targets x 5].
            predict: Tuple of predicted class and bounding box tensors.
                Class tensor is of size [batch x anchors x class]
                Bounding box tensor is of size [batch x anchors x 4].

        Returns:
            anchor_matched_targets: Tensor of size [batch x anchors x (class + 4)].
                A tensor assigning each target/gt to the best fitting anchor.
                The class probabilities are normalized.
            valid_mask: Bool tensor of shape [batch x anchors].
                True if a anchor has a target/gt assigned to it.
        """
        predict_cls, predict_bbox = predict

        # return if target has no gt information.
        n_targets = target.shape[1]
        if n_targets == 0:
            device = predict_bbox.device
            align_cls = torch.zeros_like(predict_cls, device=device)
            align_bbox = torch.zeros_like(predict_bbox, device=device)
            valid_mask = torch.zeros(predict_cls.shape[:2], dtype=bool, device=device)
            anchor_matched_targets = torch.cat([align_cls, align_bbox], dim=-1)
            return anchor_matched_targets, valid_mask

        target_cls, target_bbox = target.split([1, 4], dim=-1)  # B x N x (C B) -> B x N x C, B x N x B
        target_cls = target_cls.long().clamp(0)

        # get valid matrix (each gt appear in which anchor grid)
        grid_mask = self.get_valid_matrix(target_bbox)

        # get iou matrix (iou with each gt bbox and each predict anchor)
        iou_mat = self.get_iou_matrix(predict_bbox, target_bbox)

        # get cls matrix (cls prob with each gt class and each predict class)
        cls_mat = self.get_cls_matrix(predict_cls.sigmoid(), target_cls)

        target_matrix = (iou_mat ** self.factor["iou"]) * (cls_mat ** self.factor["cls"])

        # choose topk
        topk_targets, topk_mask = self.filter_topk(target_matrix, grid_mask, topk=self.topk)

        # match best anchor to valid targets without valid anchors
        topk_mask = self.ensure_one_anchor(target_matrix, topk_mask)

        # delete one anchor pred assign to mutliple gts
        unique_indices, valid_mask, topk_mask = self.filter_duplicates(iou_mat, topk_mask)

        align_bbox = torch.gather(target_bbox, 1, unique_indices.repeat(1, 1, 4))
        align_cls_indices = torch.gather(target_cls, 1, unique_indices)
        align_cls = torch.zeros_like(align_cls_indices, dtype=torch.bool).repeat(1, 1, self.class_num)
        align_cls.scatter_(-1, index=align_cls_indices, src=~align_cls)

        # normalize class ditribution
        iou_mat *= topk_mask
        target_matrix *= topk_mask
        metrics = (iou_mat ** self.factor["iou"]) * (cls_mat ** self.factor["cls"])
        metrics = metrics * topk_mask  # Zero out non-candidate anchors
        max_metrics = metrics.amax(dim=-1, keepdim=True).clamp_(min=1e-9)
        qual = (metrics / max_metrics).permute(0, 2, 1).gather(2, unique_indices)  # [B,A,1]
        align_cls = align_cls * qual * valid_mask[:, :, None]
        anchor_matched_targets = torch.cat([align_cls, align_bbox], dim=-1)
        return anchor_matched_targets, valid_mask


class Vec2Box:
    def __init__(self, model: YOLO, anchor_cfg: AnchorConfig, image_size, device):
        self.device = device

        if hasattr(anchor_cfg, "strides"):
            logger.info(f":japanese_not_free_of_charge_button: Found stride of model {anchor_cfg.strides}")
            self.strides = anchor_cfg.strides
        else:
            logger.info(":teddy_bear: Found no stride of model, performed a dummy test for auto-anchor size")
            self.strides = self.create_auto_anchor(model, image_size)

        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.image_size = image_size
        self.anchor_grid, self.scaler = anchor_grid.to(device), scaler.to(device)

    def create_auto_anchor(self, model: YOLO, image_size):
        W, H = image_size
        # TODO: need accelerate dummy test
        dummy_input = torch.zeros(1, 3, H, W)
        dummy_output = model(dummy_input)
        strides = []
        for predict_head in dummy_output["Main"]:
            _, _, *anchor_num = predict_head[2].shape
            strides.append(W // anchor_num[1])
        return strides

    def update(self, image_size):
        """
        image_size: W, H
        """
        if self.image_size == image_size:
            return
        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.image_size = image_size
        self.anchor_grid, self.scaler = anchor_grid.to(self.device), scaler.to(self.device)

    def __call__(self, predicts):
        preds_cls, preds_anc, preds_box = [], [], []
        for layer_output in predicts:
            pred_cls, pred_anc, pred_box = layer_output
            preds_cls.append(rearrange(pred_cls, "B C h w -> B (h w) C"))
            preds_anc.append(rearrange(pred_anc, "B A R h w -> B (h w) R A"))
            preds_box.append(rearrange(pred_box, "B X h w -> B (h w) X"))
        preds_cls = torch.concat(preds_cls, dim=1)
        preds_anc = torch.concat(preds_anc, dim=1)
        preds_box = torch.concat(preds_box, dim=1)

        pred_LTRB = preds_box * self.scaler.view(1, -1, 1)
        lt, rb = pred_LTRB.chunk(2, dim=-1)
        preds_box = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)
        return preds_cls, preds_anc, preds_box


class Anc2Box:
    def __init__(self, model: YOLO, anchor_cfg: AnchorConfig, image_size, device):
        self.device = device

        if hasattr(anchor_cfg, "strides"):
            logger.info(f":japanese_not_free_of_charge_button: Found stride of model {anchor_cfg.strides}")
            self.strides = anchor_cfg.strides
        else:
            logger.info(":teddy_bear: Found no stride of model, performed a dummy test for auto-anchor size")
            self.strides = self.create_auto_anchor(model, image_size)

        self.head_num = len(anchor_cfg.anchor)
        self.anchor_grids = self.generate_anchors(image_size)
        self.anchor_scale = tensor(anchor_cfg.anchor, device=device).view(self.head_num, 1, -1, 1, 1, 2)
        self.anchor_num = self.anchor_scale.size(2)
        self.class_num = model.num_classes

    def create_auto_anchor(self, model: YOLO, image_size):
        W, H = image_size
        dummy_input = torch.zeros(1, 3, H, W).to(self.device)
        dummy_output = model(dummy_input)
        strides = []
        for predict_head in dummy_output["Main"]:
            _, _, *anchor_num = predict_head.shape
            strides.append(W // anchor_num[1])
        return strides

    def generate_anchors(self, image_size: List[int]):
        anchor_grids = []
        for stride in self.strides:
            W, H = image_size[0] // stride, image_size[1] // stride
            anchor_h, anchor_w = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing="ij")
            anchor_grid = torch.stack((anchor_w, anchor_h), 2).view((1, 1, H, W, 2)).float().to(self.device)
            anchor_grids.append(anchor_grid)
        return anchor_grids

    def update(self, image_size):
        self.anchor_grids = self.generate_anchors(image_size)

    def __call__(self, predicts: List[Tensor]):
        preds_box, preds_cls, preds_cnf = [], [], []
        for layer_idx, predict in enumerate(predicts):
            predict = rearrange(predict, "B (L C) h w -> B L h w C", L=self.anchor_num)
            pred_box, pred_cnf, pred_cls = predict.split((4, 1, self.class_num), dim=-1)
            pred_box = pred_box.sigmoid()
            pred_box[..., 0:2] = (pred_box[..., 0:2] * 2.0 - 0.5 + self.anchor_grids[layer_idx]) * self.strides[
                layer_idx
            ]
            pred_box[..., 2:4] = (pred_box[..., 2:4] * 2) ** 2 * self.anchor_scale[layer_idx]
            preds_box.append(rearrange(pred_box, "B L h w A -> B (L h w) A"))
            preds_cls.append(rearrange(pred_cls, "B L h w C -> B (L h w) C"))
            preds_cnf.append(rearrange(pred_cnf, "B L h w C -> B (L h w) C"))

        preds_box = torch.concat(preds_box, dim=1)
        preds_cls = torch.concat(preds_cls, dim=1)
        preds_cnf = torch.concat(preds_cnf, dim=1)

        preds_box = transform_bbox(preds_box, "xycwh -> xyxy")
        return preds_cls, None, preds_box, preds_cnf.sigmoid()



class FusedONNXConverter:
    def __init__(self, class_num: int):
        self.class_num = class_num

    def update(self, image_size):
        return

    def __call__(self, fused: Tensor):
        if fused.ndim != 3:
            raise ValueError("Fused ONNX output must be 3-dimensional [B, F, N]")
        if fused.shape[1] == 4 + self.class_num:
            features = fused
        elif fused.shape[2] == 4 + self.class_num:
            features = fused.permute(0, 2, 1)
        else:
            raise ValueError("Unexpected fused tensor shape")
        cxcywh = features[:, :4, :]
        cx, cy, w, h = cxcywh.split(1, dim=1)
        half_w = w * 0.5
        half_h = h * 0.5
        x1 = cx - half_w
        y1 = cy - half_h
        x2 = cx + half_w
        y2 = cy + half_h
        boxes = torch.cat([x1, y1, x2, y2], dim=1).permute(0, 2, 1)
        classes = features[:, 4:, :].permute(0, 2, 1)
        return classes, None, boxes


def create_converter(
    model_version: str = "v9-c",
    model: Optional[YOLO] = None,
    anchor_cfg: Optional[AnchorConfig] = None,
    image_size: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    *,
    class_num: Optional[int] = None,
) -> Union[Anc2Box, Vec2Box, FusedONNXConverter]:
    if getattr(model, 'fused_onnx_output', False):
        if class_num is None:
            raise ValueError('Fused ONNX converter requires class_num')
        return FusedONNXConverter(class_num)
    if "v7" in model_version:
        return Anc2Box(model, anchor_cfg, image_size, device)
    return Vec2Box(model, anchor_cfg, image_size, device)


def bbox_nms(
    cls_dist: Tensor,              # [B, A, C] (logits or pre-probabilities)
    bbox: Tensor,                  # [B, A, 4]  (xyxy, pixel coordinates)
    nms_cfg,                       # configuration with min_confidence, min_iou, max_bbox, etc.
    confidence: Tensor = None,     # [B, A] or [B, A, 1] (objectness logits or probabilities)
):
    B, A, C = cls_dist.shape
    device = cls_dist.device
    dtype = cls_dist.dtype

    DEFAULT_PRE_TOPK = 20000
    DEFAULT_MULTI_LABEL = False
    DEFAULT_CLASS_AGNOSTIC = False
    DEFAULT_SIZE_BIAS_ALPHA = 0.0

    def _cfg_value(name: str, default):
        if nms_cfg is None:
            return default
        if hasattr(nms_cfg, name):
            return getattr(nms_cfg, name)
        if isinstance(nms_cfg, dict):
            return nms_cfg.get(name, default)
        try:
            return nms_cfg[name]
        except (TypeError, KeyError, IndexError):
            return default

    pre_topk = _cfg_value("pre_topk", DEFAULT_PRE_TOPK)
    multi_label = _cfg_value("multi_label", DEFAULT_MULTI_LABEL)
    class_agnostic = _cfg_value("class_agnostic", DEFAULT_CLASS_AGNOSTIC)
    size_bias_alpha = _cfg_value("size_bias_alpha", DEFAULT_SIZE_BIAS_ALPHA)

    # objectness (existence probability)
    if confidence is None:
        obj = torch.ones((B, A, 1), device=device, dtype=dtype)
    else:
        obj = confidence
        if obj.ndim == 2:
            obj = obj.unsqueeze(-1)
        # convert logits to probabilities in [0, 1]
        if obj.min() < 0 or obj.max() > 1:
            obj = obj.sigmoid()

    # class probabilities
    prob = cls_dist
    if prob.min() < 0 or prob.max() > 1:
        prob = prob.sigmoid()

    outputs = []
    for i in range(B):
        pi = prob[i]        # [A, C]
        bi = bbox[i]        # [A, 4]
        oi = obj[i]         # [A, 1]

        if not multi_label:
            # ---- Single-label (stable path) ----
            cls_p, cls_id = pi.max(dim=1, keepdim=True)               # [A,1], [A,1]
            score = (cls_p * oi).squeeze(1)                           # [A]

            # optional size bias
            if size_bias_alpha != 0.0:
                w = (bi[:, 2] - bi[:, 0]).clamp_(min=0)
                h = (bi[:, 3] - bi[:, 1]).clamp_(min=0)
                area = (w * h).clamp_(min=1.0)
                score = score * (area ** size_bias_alpha)

            # confidence thresholding
            keep = score > nms_cfg.min_confidence
            if keep.sum() == 0:
                outputs.append(bi.new_zeros((0, 6)))
                continue

            bi, score, cls_id = bi[keep], score[keep], cls_id[keep].squeeze(1)

            # pre-NMS top-K (sorted by score)
            if pre_topk is not None and bi.size(0) > pre_topk:
                topk_idx = score.topk(pre_topk).indices
                bi = bi.index_select(0, topk_idx)
                score = score.index_select(0, topk_idx)
                cls_id = cls_id.index_select(0, topk_idx)

            # NMS (class-wise or class-agnostic)
            ids = torch.zeros_like(cls_id) if class_agnostic else cls_id
            keep_idx = batched_nms(bi, score, ids, nms_cfg.min_iou)

            # resort by score → [:max_bbox]
            keep_idx = keep_idx[: nms_cfg.max_bbox]
            si = score.index_select(0, keep_idx)
            ci = cls_id.index_select(0, keep_idx).to(torch.int64)
            bi = bi.index_select(0, keep_idx)

            pred = torch.cat([ci[:, None].to(bi.dtype), bi, si[:, None]], dim=1)  # [N, 6]
            outputs.append(pred)

        else:
            # ---- Multi-label (only when requested) ----
            # enumerate (anchor, class) pairs above the threshold
            score_mat = pi * oi                                  # [A, C]
            ai, ci = torch.where(score_mat > nms_cfg.min_confidence)
            if ai.numel() == 0:
                outputs.append(bi.new_zeros((0, 6)))
                continue

            si = score_mat[ai, ci]                               # [N]
            b2 = bi[ai]                                          # [N, 4]
            # size bias
            if size_bias_alpha != 0.0:
                w = (b2[:, 2] - b2[:, 0]).clamp_(min=0)
                h = (b2[:, 3] - b2[:, 1]).clamp_(min=0)
                area = (w * h).clamp_(min=1.0)
                si = si * (area ** size_bias_alpha)

            # pre-NMS top-K
            if pre_topk is not None and si.size(0) > pre_topk:
                topk_idx = si.topk(pre_topk).indices
                b2 = b2.index_select(0, topk_idx)
                si = si.index_select(0, topk_idx)
                ci = ci.index_select(0, topk_idx)

            ids = torch.zeros_like(ci) if class_agnostic else ci
            keep_idx = batched_nms(b2, si, ids, nms_cfg.min_iou)

            # resort by score → [:max_bbox]
            keep_idx = keep_idx[: nms_cfg.max_bbox]
            out = torch.cat([ci.index_select(0, keep_idx)[:, None].to(b2.dtype),
                             b2.index_select(0, keep_idx),
                             si.index_select(0, keep_idx)[:, None]], dim=1)
            outputs.append(out)

    return outputs


def calculate_map(predictions, ground_truths) -> Dict[str, Tensor]:
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    mAP = metric([to_metrics_format(predictions)], [to_metrics_format(ground_truths)])
    return mAP


def to_metrics_format(prediction: Tensor) -> Dict[str, Union[float, Tensor]]:
    prediction = prediction[prediction[:, 0] != -1]
    bbox = {"boxes": prediction[:, 1:5], "labels": prediction[:, 0].int()}
    if prediction.size(1) == 6:
        bbox["scores"] = prediction[:, 5]
    return bbox
