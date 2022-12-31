import torch
from torch import nn


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion

        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            if losses.mean()<0:
                print(losses.mean())
            return losses.mean()

class GIOU(nn.Module):
    def __init__(self,):
        super(GIOU, self).__init__()

    def forward(self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
        https://arxiv.org/abs/1902.09630

        Gradient-friendly IoU loss with an additional penalty that is non-zero when the
        boxes do not overlap and scales with the size of their smallest enclosing box.
        This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

        Args:
            boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
            reduction: 'none' | 'mean' | 'sum'
                     'none': No reduction will be applied to the output.
                     'mean': The output will be averaged.
                     'sum': The output will be summed.
            eps (float): small number to prevent division by zero
        """

        x1, y1, x2, y2 = boxes1.unbind(dim=-1)
        x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

        assert (x2 >= x1).all(), "bad box: x1 larger than x2"
        assert (y2 >= y1).all(), "bad box: y1 larger than y2"

        # Intersection keypoints
        xkis1 = torch.max(x1, x1g)
        ykis1 = torch.max(y1, y1g)
        xkis2 = torch.min(x2, x2g)
        ykis2 = torch.min(y2, y2g)

        intsctk = torch.zeros_like(x1)
        mask = (ykis2 > ykis1) & (xkis2 > xkis1)
        intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
        iouk = intsctk / (unionk + eps)

        # smallest enclosing box
        xc1 = torch.min(x1, x1g)
        yc1 = torch.min(y1, y1g)
        xc2 = torch.max(x2, x2g)
        yc2 = torch.max(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1)
        miouk = iouk - ((area_c - unionk) / (area_c + eps))

        loss = 1 - miouk

        if reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

class iou_loss(nn.Module):
    def __init__(self,):
        super(iou_loss, self).__init__()

    def forward(self, preds, bbox, eps=1e-6, reduction='mean'):
        '''
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/iou_loss.py
        :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        :return: loss
        '''
        x1 = torch.max(preds[:, 0], bbox[:, 0])
        y1 = torch.max(preds[:, 1], bbox[:, 1])
        x2 = torch.min(preds[:, 2], bbox[:, 2])
        y2 = torch.min(preds[:, 3], bbox[:, 3])

        w = (x2 - x1 + 1.0).clamp(0.)
        h = (y2 - y1 + 1.0).clamp(0.)

        inters = w * h

        uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
                bbox[:, 3] - bbox[:, 1] + 1.0) - inters

        ious = (inters / uni).clamp(min=eps)
        # loss = -ious.log()
        loss=1-ious

        if reduction == 'mean':
            loss = torch.mean(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)
        else:
            raise NotImplementedError
        return loss

linear_iou = IOULoss(loc_loss_type='linear_iou')
giou=GIOU()
iouloss=iou_loss()
