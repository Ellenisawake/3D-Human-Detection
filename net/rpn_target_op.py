from net.common import *
from net.configuration import *
from net.processing.boxes import *
from net.blocks import *
from net.utility.draw import *

## base box ##

def convert_w_h_cx_cy(base):
    """ Return width, height, x center, and y center for a base (box). """

    w  = base[2] - base[0] + 1
    h  = base[3] - base[1] + 1
    cx = base[0] + 0.5 * (w - 1)
    cy = base[1] + 0.5 * (h - 1)
    return w, h, cx, cy


def make_bases_given_ws_hs(ws, hs, cx, cy):
    """ Given a vector of widths (ws) and heights (hs) around a center(cx, cy), output a set of bases. """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    bases = np.hstack((cx - 0.5 * (ws - 1),
                       cy - 0.5 * (hs - 1),
                       cx + 0.5 * (ws - 1),
                       cy + 0.5 * (hs - 1)))
    return bases


def make_bases_given_ratios(base, ratios):
    """  Enumerate a set of bases for each aspect ratio wrt a base.  """

    w, h, cx, cy = convert_w_h_cx_cy(base)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    bases = make_bases_given_ws_hs(ws, hs, cx, cy)
    return bases


def make_bases_given_scales(base, scales):
    """ Enumerate a set of  bases for each scale wrt a base. """

    w, h, cx, cy = convert_w_h_cx_cy(base)
    ws = w * scales
    hs = h * scales
    bases = make_bases_given_ws_hs(ws, hs, cx, cy)
    return bases


def make_bases(
    base_size = 16,
    ratios=[0.5, 1, 2],
    scales=2**np.arange(3, 6)):

    """  Generate bases by enumerating aspect ratios * scales, wrt a reference (0, 0, 15, 15)  base (box). """

    base        = np.array([1, 1, base_size, base_size]) - 1
    ratio_bases = make_bases_given_ratios(base, ratios)
    bases = np.vstack(
        [make_bases_given_scales(ratio_bases[i, :], scales) for i in range(ratio_bases.shape[0])])
    return bases



# ## rpn layer op ##
# def subset_to_set (data, count, inds, fill=0):
#     ''' Unmap a subset of item (data) back to the original set of items (of size count) '''
#
#     if len(data.shape) == 1:
#         ret = np.empty((count, ), dtype=data.dtype)
#         ret.fill(fill)
#         ret[inds] = data
#     else:
#         ret = np.empty((count, ) + data.shape[1:], dtype=data.dtype)
#         ret.fill(fill)
#         ret[inds, :] = data
#
#     return ret




def make_anchors(bases, stride, image_shape, feature_shape, allowed_border=0):

    H, W                  = feature_shape
    img_height, img_width = image_shape

    # anchors = shifted bases. Generate proposals from box deltas and anchors
    shift_x = np.arange(0, W) * stride
    shift_y = np.arange(0, H) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    B  = len(bases)
    HW = len(shifts)
    anchors   = (bases.reshape((1, B, 4)) + shifts.reshape((1, HW, 4)).transpose((1, 0, 2)))
    anchors   = anchors.reshape((HW * B, 4)).astype(np.int32)
    num_anchors = int(HW * B)

    # only keep anchors inside the image
    inside_inds = np.where(
        (anchors[:, 0] >= -allowed_border) &
        (anchors[:, 1] >= -allowed_border) &
        (anchors[:, 2] < img_width  + allowed_border) &  # width
        (anchors[:, 3] < img_height + allowed_border)    # height
    )[0].astype(np.int32)

    return anchors, inside_inds





# gt_boxes : (x1, y1, x2, y2)
def rpn_target( anchors, inside_inds, gt_labels,  gt_boxes):

    inside_anchors = anchors[inside_inds, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inside_inds), ), dtype=np.int32)
    labels.fill(-1)

    # overlaps between the anchors and the gt process
    overlaps = box_overlaps(
        np.ascontiguousarray(inside_anchors,  dtype=np.float),
        np.ascontiguousarray(gt_boxes,        dtype=np.float))

    argmax_overlaps    = overlaps.argmax(axis=1)
    max_overlaps       = overlaps[np.arange(len(inside_inds)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps    = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    labels[max_overlaps <  CFG.TRAIN.RPN_BG_THRESH_HI] = 0   # bg label
    labels[gt_argmax_overlaps] = 1                           # fg label: for each gt, anchor with highest overlap
    labels[max_overlaps >= CFG.TRAIN.RPN_FG_THRESH_LO] = 1   # fg label: above threshold IOU


    # subsample positive labels
    num_fg = int(CFG.TRAIN.RPN_FG_FRACTION * CFG.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = np.random.choice( fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels
    num_bg = CFG.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    idx_label  = np.where(labels != -1)[0]
    idx_target = np.where(labels ==  1)[0]

    inds   = inside_inds[idx_label]
    labels = labels[idx_label]

    pos_inds = inside_inds[idx_target]
    pos_anchors  = inside_anchors[idx_target]
    pos_gt_boxes = (gt_boxes[argmax_overlaps])[idx_target]
    targets = box_transform(pos_anchors, pos_gt_boxes)

    return inds, pos_inds, labels, targets



# def tf_rpn_target(gt_boxes, anchors, inside_inds, name='rpn_target'):
#
#     #<todo>
#     #assert batch_size == 1, 'Only single image batches are supported'
#
#     return  \
#         tf.py_func(rpn_target,[gt_boxes, anchors, inside_inds], [tf.int32, tf.float32, tf.float32],
#         name =name)
#


## unit test ##---
def draw_rpn_gt(image, gt_boxes, gt_labels=None, darken=0.7):

    ## gt
    img_gt = image.copy()*darken
    num =len(gt_boxes)
    for n in range(num):
        b = gt_boxes[n]
        cv2.rectangle(img_gt,(b[0],b[1]),(b[2],b[3]),(0,255,255),2)

    return img_gt


def draw_rpn_labels(image, anchors, inds, labels, darken=0.7):

    is_print=0
    ## yellow (thick): gt
    ## red     : regression target (before)
    ## yellow  : regression target (after)
    ## red  + dot : +ve label
    ## grey + dot : -ve label

    ## draw +ve/-ve labels ......
    num_anchors = len(anchors)
    labels = labels.reshape(-1)

    fg_label_inds = inds[np.where(labels == 1)[0]]
    bg_label_inds = inds[np.where(labels == 0)[0]]
    num_pos_label = len(fg_label_inds)
    num_neg_label = len(bg_label_inds)
    if is_print: print ('rpn label : num_pos=%d num_neg=%d,  all = %d'  %(num_pos_label, num_neg_label,num_pos_label+num_neg_label))

    img_label = image.copy()*darken
    for i in bg_label_inds:
        a = anchors[i]
        cv2.rectangle(img_label,(a[0], a[1]), (a[2], a[3]), (32,32,32), 1)
        cv2.circle(img_label,(a[0], a[1]),2, (32,32,32), -1)

    for i in fg_label_inds:
        a = anchors[i]
        cv2.rectangle(img_label,(a[0], a[1]), (a[2], a[3]), (0,0,255), 1)
        cv2.circle(img_label,(a[0], a[1]),2, (0,0,255), -1)

    return img_label




def draw_rpn_targets(image, anchors, pos_inds, targets, darken=0.7):
    is_print=0

    #draw +ve targets ......
    fg_target_inds = pos_inds
    num_pos_target = len(fg_target_inds)
    if is_print: print ('rpn target : num_pos=%d'  %(num_pos_target))

    img_target = image.copy()*darken
    for n,i in enumerate(fg_target_inds):
        a = anchors[i]
        t = targets[n]
        b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
        b = b.reshape(-1).astype(np.int32)

        cv2.rectangle(img_target,(a[0], a[1]), (a[2], a[3]), (0,0,255), 1)
        cv2.rectangle(img_target,(b[0], b[1]), (b[2], b[3]), (0,255,255), 1)
    return img_target




###  <todo> this function has not been updated
# def test_op():
#
#
#     if 0:
#         bases = make_bases(
#             base_size = 16,
#             #ratios=[0.5, 1, 2],
#             #scales=2**np.arange(3, 6))
#             ratios=[0.5, 1, 2],
#             scales=2**np.arange(3, 4 ))
#
#         #(x1, y1, x2, y2, label)
#         # gt_boxes = np.array([[
#         #     150,100,300,340, 1
#         # ]])
#         num_bases = len(bases)
#         stride = 16
#         image_shape   = (480,640,3)
#         feature_shape = (30,40,64)
#
#
#     if 1:
#         ratios=np.array([0.5,1,2], dtype=np.float32)
#         scales=np.array([1,2,3],   dtype=np.int32)
#         bases = make_bases(
#             base_size = 16,
#             ratios=ratios,
#             scales=scales
#         )
#         num_bases = len(bases)
#         stride = 16
#         image_shape   = (400, 400, 3)
#         feature_shape = (25, 25,  2*num_bases)
#
#
#     if 1:
#         gt_boxes  = np.load('/root/share/project/didi/data/kitti/dummy/one_frame/gt_boxes.npy')
#         num_gts = len(gt_boxes)
#
#
#     if 0:
#         num_gts = 14
#         np.random.seed(101)
#         gt_boxes = np.zeros(shape=(num_gts,5),dtype=np.int)
#         num_bases = len(bases)
#         for n in range(num_gts):
#             i = np.random.randint(0,num_bases)
#             b = bases[i]
#             w = b[2]-b[0]+1
#             h = b[3]-b[1]+1
#             height, width,_ = image_shape
#
#
#             x1=np.random.randint(0,max(1,width -w))
#             y1=np.random.randint(0,max(1,height-h))
#             x2=x1+w
#             y2=y1+h
#             gt_boxes[n] = np.array([x1,y1,x2,y2,1])
#
#
#     image = np.random.uniform(0,255,size=(1, *image_shape))
#     for n in range(num_gts):
#         b = gt_boxes[n]
#         cv2.rectangle(image[0],(b[0],b[1]),(b[2],b[3]),(128,128,128 ),-1)
#
#
#     feature = np.random.uniform(0,1,size=(1, *feature_shape))
#     batch_size, img_height, img_width, _ = image.shape
#
#
#     anchors, inside_inds =  make_anchors(bases, stride, image_shape, feature_shape)
#
#
#     ##check python code without tf ------------------
#     if 0:
#         rpn_labels, rpn_box_targets, rpn_box_weights = rpn_target(gt_boxes, anchors, inside_inds)
#
#         #show
#         print ('** without tf **')
#         norm_feature = normalise(feature[0,:,:,0:3])
#         norm_image = normalise(image[0,:,:,:])
#         imshow('feature', norm_feature, resize=stride)
#         draw_op_results(rpn_labels,rpn_box_targets,rpn_box_weights,anchors,norm_image,gt_boxes)
#         cv2.waitKey(0)
#         exit(0)
#     ##-----------------------------------------------
#
#
#     rpn_gt_boxes    = tf.placeholder(shape=[None, 5], dtype=tf.float32, name ='gt_boxes'   )  #gt_boxes
#     rpn_anchors     = tf.placeholder(shape=[None, 4], dtype=tf.int32,   name ='anchors'    )  #anchors
#     rpn_inside_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name ='inside_inds')  #inside_inds
#     rpn_target1 = tf_rpn_target (rpn_gt_boxes, rpn_anchors, rpn_inside_inds)
#
#
#     with tf.Session()  as sess:
#         sess.run(tf.global_variables_initializer(),{IS_TRAIN_PHASE:True})
#         #summary_writer = tf.summary.FileWriter(out_dir, sess.graph)
#
#         fd = {
#             rpn_gt_boxes:    gt_boxes,
#             rpn_anchors :    anchors,
#             rpn_inside_inds: inside_inds,
#             IS_TRAIN_PHASE: True
#         }
#         rpn_labels, rpn_box_targets, rpn_box_weights = sess.run(rpn_target1,fd)
#
#
#         #show
#         print ('** tf **')
#         norm_feature = normalise(feature[0,:,:,0:3])
#         norm_image = normalise(image[0,:,:,:])
#         imshow('feature', norm_feature, resize=stride)
#         draw_op_results(rpn_labels,rpn_box_targets,rpn_box_weights,anchors,norm_image,gt_boxes)
#         cv2.waitKey(0)
#         exit(0)
#
#         pass


## main ##----------------------------------------------------------------
if __name__ == '__main__':
    print  ('\"%s\" running main function ...' % os.path.basename(__file__))

    test_op()