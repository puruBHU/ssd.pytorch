import torch
from torch.autograd import Function
from box_utils import decode, nms
from config import voc as cfg
import numpy as np
from SSD_generate_anchors import generate_ssd_priors
import collections

class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output

if __name__=='__main__':
    
    SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])

    Spec = collections.namedtuple('Spec', ['feature_map_size', 'shrinkage', 'box_sizes', 
                                           'aspect_ratios'])

# the SSD orignal specs
    specs = [
        Spec(38, 8, SSDBoxSizes(30, 60), [2]),
        Spec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
        Spec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
        Spec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
        Spec(3, 100, SSDBoxSizes(213, 264), [2]),
        Spec(1, 300, SSDBoxSizes(264, 315), [2])
    ]

    priors = generate_ssd_priors(specs)
    priors_th  = torch.Tensor(priors)
#    priors_th.unsqueeze_(0)
    
    loc_data = np.random.randn(4, 8732 , 4)
    conf_data = np.random.randint(low=0, high =1, size=(4 , 8732, 21), dtype=np.int16)
    
    loc_data_th = torch.Tensor(loc_data)
    conf_data_th = torch.Tensor(conf_data).view(-1, 21)
    
    
    hello = Detect(num_classes=21, bkg_label=None, top_k=200, conf_thresh=0.5, nms_thresh=0.6)
    test   = hello.forward(loc_data= loc_data_th, conf_data=conf_data_th, prior_data=priors_th)