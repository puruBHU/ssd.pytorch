#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:10:11 2019

@author: Purnendu Mishra
"""
from keras.layers import Conv2D, SeparableConv2D, Activation
from keras.layers import BatchNormalization
from keras.initializers import he_normal
from keras.regularizers import l2
from keras import backend as K

import numpy as np

def _bn_relu(input_):
    norm = BatchNormalization()(input_)
    return Activation('relu')(norm)

def conv_bn_relu(**params):
    filters     = params['filters']
    kernel_size = params.setdefault('kernel_size', (3,3))
    strides     = params.setdefault('strides',(1,1))
    padding     = params.setdefault('padding','same')
    dilation_rate = params.setdefault('dilation_rate', 1)
    kernel_initializer = params.setdefault('kernel_initializer', he_normal())
    kernel_regularizer = params.setdefault('kernel_regularizer', l2(1e-3))
    activation         = params.setdefault('activation','relu')
    name               = params.setdefault('name', None)

    def f(input_):
        conv = Conv2D(filters       = filters,
                      kernel_size   = kernel_size,
                      strides       = strides,
                      padding       = padding,
                      dilation_rate = dilation_rate,
                      kernel_initializer = kernel_initializer,
                      kernel_regularizer = kernel_regularizer,
                      name = name['conv'])(input_)

        batch_norm = BatchNormalization(name = name['batch_norm'])(conv)

        return Activation(activation,name = name['activation'])(batch_norm)
    return f


def point_form(boxes):
    """ Convert prior boxes to (xmin, ymin, xmax, ymax)
    """
    top    = boxes[:,:2] - boxes[:,2:] /2
    bottom = boxes[:,:2] + boxes[:,2:]/2

    return np.concatenate((top, bottom), axis=1)

def center_form(boxes):
    """ Convert prior boxes to (cx, cy, w, h)
    
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    """
    center_coordinates = (boxes[:,:2] + boxes[:,2:]) / 2
    width_hegight      = (boxes[:,2:] - boxes[:,:2]) / 2
    
    return np.concatenate((center_coordinates, width_hegight), axis=1)


def intersect(box_a, box_b):

    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    box_a = np.array(box_a, dtype = np.float32)
    box_b = np.array(box_b, dtype = np.float32) 
    
    A = box_a.shape[0]
    B = box_b.shape[0]
    

    
    min_xy = np.maximum(box_a[:,:2].reshape(A, 1, -1).repeat(B, axis = 1),
                        box_b[:,:2].reshape(1, B, -1).repeat(A, axis = 0))
    
   
    max_xy = np.minimum(box_a[:,2:].reshape(A, 1, -1).repeat(B, axis = 1),
                        box_b[:,2:].reshape(1, B, -1).repeat(A, axis = 0))
    
    inter = np.clip((max_xy - min_xy), a_min = 0, a_max = None)
    
    return inter[:,:,0] * inter[:,:,1]
   


def jaccard(box_a, box_b):
    
    intersection = intersect(box_a, box_b)

    A = box_a.shape[0]
    B = box_b.shape[0]
    

    
    # area_box_a = (xmax - xmin) * (ymax - ymin)
    area_box_a  = (box_a[:,2] - box_a[:,0]) * (box_a[:,3] - box_a[:,1])
    
    area_box_a = area_box_a.reshape(-1, 1)
    area_box_a = area_box_a.repeat(B, axis = 1)
    
    # calculate areas of box B
    area_box_b  = (box_b[:,2] - box_b[:,0]) * (box_b[:,3] - box_b[:,1])
    
    area_box_b = area_box_b.reshape(1,-1)
    area_box_b = area_box_b.repeat(A, axis = 0)
    
    union       = area_box_a + area_box_b - intersection
    
    iou          = intersection/ union
    
    return iou

def match(truths      = None, 
          labels     = None, 
          priors     = None, 
          variance   = None, 
          threshold  = 0.5,
#          idx        = None,
#          loc_t      = None,
#          conf_t     = None
        ):
    """
    Match each prior (or anchor) box with the ground truth box of the ighest jaccard overlap, 
    encode the bounding boxes, then return the matched indices correspoding to both confidence 
    and location predictions.
    
    Both Truth and Priors are in the form (cx, cy, w, h)
    Convert to form (xmin, ymin,xmax, ymax) before getting IOU
    
    Arguments:
        threshold: (float) The overlap threshold used when matching boxes
        truth    : (tensor) Ground truth boxes, sahep [num_obj, num_priors]
        priors   : (tensor)  Prior boxes from prior boxes layers, shape [num_prioirs, 4]
        variance : (tensor) Variance corresponding to each prioir coordinate, shape [num_priors, 4]
        
        labels   : (tensor) All the class label for the image, shape : [num_obj]
        
        loc_t    : (tensor) Tensor to be filled with encoded location targets
        conf_t   : (tensor) Tensor to be filled with ematched indices for conf preds.
        idx     : (int) current batch index
        
    Returns:
        The match indices corresponding to 
            1) location 
            2) cofidence predcition
    """
    # Ground truth are  in form (xc, yc, w, h)
    # convert it to form (xmin, ymin, xmax, ymax)
    truths = point_form(truths)
    
    iou = jaccard(truths, point_form(priors))
    
    best_prior_overlap = np.amax(iou, axis=-1).astype(np.float32)
    best_prior_idx     = np.argmax(iou, axis =-1)
    
#    print(best_prior_overlap.shape)
#    print(best_prior_idx.shape)

    best_truth_overlap = np.amax(iou, axis=0).astype(np.float32)
    best_truth_idx     = np.argmax(iou, axis = 0)
    
#    print(best_truth_overlap.shape)
#    print(best_truth_idx.shape)

    for j in range(best_prior_idx.shape[0]):
        best_truth_idx[best_prior_idx[j]] = j
    
    matches = truths[best_truth_idx]
    conf    = labels[best_truth_idx]

    conf[best_truth_overlap < threshold] = 0
    
    loc       = encode(matched=matches, priors=priors, variances=variance)
    
    
    
    return loc, conf
    

def encode_np(matched = None, priors = None, variances = [0.1, 0.2]):
    '''
    Encode the variance from the priorbox layers inot the ground truth boxes 
    we have macthed  (based on jaccard overlap) with the prior boxes
    Args:
        matched:  (tensor) coords of ground truth for each prior in point_form 
                   shape = [num_priors, 4]
       priors  : (tensor) priors boxes in center-offset form 
                   shape = [num_priors, 4]
       variance: list(float) Variance of prior boxes

    Returns:
        encoded boxes: (tensor) shape = [num_priors, 4]
    '''
    g_cxcy = (matched[:,:2] + matched[:,:2])/2 - priors[:,:2]
    
    # encoded variance
    g_cxcy /= (variances[0] * priors[:,2:])
    
    # match width/height with priors width and height
    g_wh  = (matched[:,2:] - matched[:,:2]) / priors[:,2:]
    g_wh  = np.log(g_wh) / variances[1]
    
    return np.concatenate((g_cxcy, g_wh), axis = -1)

def decode_np(loc = None, priors=None, variances = [0.1, 0.2]):
    xy = priors[:,:2] + loc[:,:2] * variances[0] * priors[:,:2]
    wh = priors[:,2:] * np.exp(loc[:,2:] * variances[1])
    
    boxes = np.concatenate((xy, wh), axis=-1)
    
    boxes[:,:2] -= boxes[:,2:] / 2
    boxes[:,2:] += boxes[:,:2]
    return boxes

def log_sum_exp(x):
    x_max = np.max(x)
    return np.log(np.sum(np.exp(x - x_max), axis=1, keepdims=True)) + x_max

def gather(self, dim, index):
    """
    Gathers values along an axis specified by ``dim``.

    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    Parameters
    ----------
    dim:
        The axis along which to index
    index:
        A tensor of indices of elements to gather

    Returns
    -------
    Output Tensor
    """
    idx_xsection_shape = index.shape[:dim] + \
        index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(self, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)



def non_maximum_supression(boxes, scores, overlap = 0.5, top_k= 200):

    keep = np.zeros(shape = scores.shape[0], dtype = np.float32)
    
    if len(boxes) == 0:
        return keep
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    area = (x2 - x1) * (y2 - y1)
    
    idx = np.argsort(scores)
    
    idx = idx[-top_k:]
    

    
    count = 0 
    
    while len(idx) > 0:
        i = idx[-1]  # index of current largest val
        
        keep[count] = i
        count += 1
        
        if idx.shape[0] == 1:
            break
            
        idx = idx[:-1]
        
        xx1 = np.take(x1, indices=idx, axis=0)
        yy1 = np.take(y1, indices=idx, axis=0)
        xx2 = np.take(x2, indices=idx, axis=0)
        yy2 = np.take(y2, indices=idx, axis=0)
        
        xx1 = np.clip(xx1, a_min = x1[i],  a_max=None)
        yy1 = np.clip(yy1, a_min = y1[i],  a_max=None)
        xx2 = np.clip(xx2, a_min = None,   a_max=x2[i])
        yy2 = np.clip(yy2, a_min = None,   a_max=x2[i])
        
        w = xx2 - xx1
        h = yy2 - yy1

        
        w = np.clip(w, a_min = 0., a_max = None)
        h = np.clip(h, a_min = 0., a_max = None)
        
        inter = w * h
        rem_areas = np.take(area, indices = idx, axis = 0) # load remaining areas
        union     = (rem_areas - inter) + area[i]
        IOU       = inter/union
        idx       = idx[np.less_equal(IOU, overlap)]
#         print(idx.shape)
    return keep, count
