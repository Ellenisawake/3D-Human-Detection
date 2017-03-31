from net.common import *
from net.utility.file import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *

from dummynet import *
from data import *

from net.rpn_loss_op import *
from net.rcnn_loss_op import *
from net.rpn_target_op import make_bases, make_anchors, rpn_target
from net.rcnn_target_op import rcnn_target

from net.rpn_nms_op     import draw_rpn_nms, draw_rpn
from net.rcnn_nms_op    import rcnn_nms, draw_rcnn_nms_rgb, draw_rcnn_nms_top, draw_rcnn
from net.rpn_target_op  import draw_rpn_gt, draw_rpn_targets, draw_rpn_labels
from net.rcnn_target_op import draw_rcnn_targets, draw_rcnn_labels


#http://3dimage.ee.tsinghua.edu.cn/cxz
# "Multi-View 3D Object Detection Network for Autonomous Driving" - Xiaozhi Chen, CVPR 2017

ROOT_DIR = '../3d_human_detection'

def load_dummy_datas():

    num_frames = 1
    rgbs      =[]
    lidars    =[]
    tops      =[]
    fronts    =[]
    gt_labels =[]
    gt_boxes3d=[]

    top_images  =[]
    front_images=[]

    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
    for n in range(num_frames):
        print ( 'reading %d'%n, end='\r', flush=True)

        front = np.zeros((1,1),dtype=np.float32)
        front_image = np.zeros((1,1,3),dtype=np.float32)

        lidar = np.load(ROOT_DIR + '/data/%05d.npy'%n)
        gt_box3d = np.load(ROOT_DIR + '/data/gt_boxes3d_%05d.npy'%n)
        gt_label = np.load(ROOT_DIR + '/data/gt_labels_%05d.npy'%n)


        top_image  = cv2.imread(ROOT_DIR + '/data/top_image_%05d.png'%n,1)
        top  = np.load(ROOT_DIR + '/data/top_%05d.npy'%n)
        rgb = cv2.imread(ROOT_DIR + '/data/rgb_%05d.png'%n,1)

        rgbs.append(rgb)
        lidars.append(lidar)
        tops.append(top)
        fronts.append(front)
        gt_labels.append(gt_label)
        gt_boxes3d.append(gt_box3d)

        top_images.append(top_image)
        front_images.append(front_image)


        # explore dataset:
        #print (gt_box3d)
        if 1:
            projections=box3d_to_rgb_projections(gt_box3d)
            rgb1 = draw_rgb_projections(rgb, projections, color=(255,255,255), thickness=2)
            top_image1 = draw_box3d_on_top(top_image, gt_box3d, color=(255,255,255), thickness=2)

            imshow('rgb',rgb1)
            imshow('top_image',top_image1)

            mlab.clf(fig)
            draw_lidar(lidar, fig=fig)
            draw_gt_boxes3d(gt_box3d, fig=fig)
            mlab.show(1)
            cv2.waitKey(1)

            pass


    ##exit(0)
    mlab.close(all=True)
    return  rgbs, tops, fronts, gt_labels, gt_boxes3d, top_images, front_images, lidars


#<todo>
def project_to_roi3d(top_rois):
    num = len(top_rois)
    rois3d = np.zeros((num,8,3))
    rois3d = top_box_to_box3d(top_rois[:,1:5])

    return rois3d


def project_to_rgb_roi(rois3d):
    num  = len(rois3d)
    rois = np.zeros((num,5),dtype=np.int32)
    projections = box3d_to_rgb_projections(rois3d)
    for n in range(num):
        qs = projections[n]
        minx = np.min(qs[:,0])
        maxx = np.max(qs[:,0])
        miny = np.min(qs[:,1])
        maxy = np.max(qs[:,1])
        rois[n,1:5] = minx,miny,maxx,maxy
    return rois


def box3d_to_rgb_projections1(boxes3d, Mt=None, Kt=None):

    if Mt is None: Mt = np.array(MATRIX_Mt)
    if Kt is None: Kt = np.array(MATRIX_Kt)

    num  = len(boxes3d)
    projections = np.zeros((num,8,2),  dtype=np.int32)
    for n in range(num):
        box3d0 = boxes3d[n,:]

        box3d=np.zeros((8,3),dtype=np.float)
        box3d[:,0] =   box3d0[:,0]-8
        box3d[:,1] =  -box3d0[:,2]+2.3
        box3d[:,2] =   box3d0[:,1]

        Ps = np.hstack(( box3d, np.ones((8,1))) )

        '''

        # The format of each camera is as follows:
        # Filename (of the undistorted image in visualize folder)
        # Original filename
        # Focal Length (of the undistorted image)
        # 2-vec Principal Point (image center)
        # 3-vec Translation T (as in P = K[R T])
        # 3-vec Camera Position C (as in P = K[R -RC])
        # 3-vec Axis Angle format of R
        # 4-vec Quaternion format of R
        # 3x3 Matrix format of R
        # [Normalized radial distortion] = [radial distortion] * [focal length]^2
        # 3-vec Lat/Lng/Alt from EXIF


        00000011.jpg
        /root/Desktop/soccer_data/frame_0000/soccer_game_goal-cam09-f000000.jpg
        3362.23413086
        960 540

        6.23505115509 -1.76612854004 7.38259124756


        -7.33398389816 0.0259736776352 -6.53527641296

        0.245259513006 -0.15640001546 -0.00311330440474
        0.989441044098 0.122197837847 -0.0779245766826 -0.00155116946191

        0.987851500511 -0.015974810347 -0.154582649469
        -0.0221139732748 0.970131576061 -0.241573318839
        0.153824448586 0.242056906223 0.957991242409

        -0.298312767847
        0 0 0
        '''

        M=np.array([
           [ 0.987851500511, -0.015974810347, -0.154582649469,   6.23505115509],
           [-0.0221139732748, 0.970131576061, -0.241573318839,  -1.76612854004],
           [ 0.153824448586, 0.242056906223, 0.957991242409,     7.38259124756],
           [0,0,0,1],
        ])
        Mt=M.transpose()

        K=np.array([
              [3362.23413086,    0.    ,    960    ],
              [   0.    ,  3362.23413086,   540.    ],
              [ 0,  0 ,    1.    ]
        ])
        Kt=K.transpose()

        print(Kt)
        print(Mt)


        Qs = np.matmul(Ps,Mt)
        Qs = Qs[:,0:3]
        qs = np.matmul(Qs,Kt)
        zs = qs[:,2].reshape(8,1)
        qs = (qs/zs)
        projections[n] = qs[:,0:2]

    return projections


def  project_to_front_roi(rois3d):
    num  = len(rois3d)
    rois = np.zeros((num,5),dtype=np.int32)
    return rois


#-----------------
#testing
def do_test(
    tops, fronts, rgbs, anchors, inside_inds,
    top_images,front_images,rgb_images,

    top_anchors,top_inside_inds,proposals,proposal_scores,
    top_features,top_rois,front_rois,rgb_rois,fuse_probs,fuse_deltas,
    sess,
    out_dir):

    num_frames  = len(tops)
    top_shape   = tops[0].shape
    front_shape = fronts[0].shape
    rgb_shape   = rgbs[0].shape

    for n in range(num_frames):
        batch_top_images    = tops[n].reshape(1,*top_shape)
        batch_front_images  = fronts[n].reshape(1,*front_shape)
        batch_rgb_images    = rgbs[n].reshape(1,*rgb_shape)

        ## run propsal generation ----------------
        fd1={
            top_images:      batch_top_images,
            top_anchors:     anchors,
            top_inside_inds: inside_inds,
            IS_TRAIN_PHASE:  False
        }
        batch_proposals, batch_proposal_scores, batch_top_features = sess.run([proposals, proposal_scores, top_features],fd1)
        batch_top_rois   = batch_proposals
        batch_rois3d	 = project_to_roi3d    (batch_top_rois)
        batch_front_rois = project_to_front_roi(batch_rois3d  )
        batch_rgb_rois   = project_to_rgb_roi  (batch_rois3d  )

        ## run classification and regression -----
        fd2={
            **fd1,
            top_images:   batch_top_images,
            front_images: batch_front_images,
            rgb_images:   batch_rgb_images,
            top_rois:     batch_top_rois,
            front_rois:   batch_front_rois,
            rgb_rois:     batch_rgb_rois,
        }
        batch_fuse_probs, batch_fuse_deltas = sess.run([ fuse_probs, fuse_deltas ],fd2)
        probs, boxes3d = rcnn_nms(batch_fuse_probs, batch_fuse_deltas, batch_rois3d, threshold=0.5)

        ## show rcnn(fuse) nms
        img_rcnn     = draw_rcnn (top_image, batch_fuse_probs, batch_fuse_deltas, batch_top_rois, batch_rois3d,darker=1)
        img_rcnn_nms = draw_rcnn_nms(rgb, boxes3d, probs)
        imshow('img_rcnn',img_rcnn)
        imshow('img_rcnn_nms',img_rcnn_nms)
        cv2.imwrite(out_dir+'/final/rgb_box3d%05d.png'%n,img_rcnn_nms)
        cv2.imwrite(out_dir+'/final/top_all%05d.png'%n,img_rcnn)
        np.save(out_dir+'/final/boxes3d%05d.npy'%n,boxes3d)
        np.save(out_dir+'/final/probs%05d.npy'%n,probs)
        cv2.waitKey(1)

        top_image = top_imgs[idx]
        rgb       = rgbs[idx]
        img_rgb_rois = draw_boxes(rgb, batch_rgb_rois[:,1:5], color=(255,0,255), thickness=1)
        imshow('img_rgb_rois',img_rgb_rois)
        cv2.imwrite(out_dir+'/final/rgb_proposals%05d.png'%n,img_rgb_rois)

    pass


def run_train():
    out_dir = ROOT_DIR + '/out'

    makedirs(out_dir +'/tf')
    makedirs(out_dir +'/check_points')
    log = Logger(out_dir+'/log.txt',mode='a')

    initial_model=None

    if 1:
        ratios=np.array([1], dtype=np.float32)
        scales=np.array([1,2],   dtype=np.float32)
        bases = make_bases(
            base_size = 16,
            ratios=ratios,
            scales=scales
        )
        num_bases = len(bases)
        stride = 8

        rgbs, tops, fronts, gt_labels, gt_boxes3d, top_imgs, front_imgs, lidars = load_dummy_datas()
        num_frames = len(rgbs)

        top_shape   = tops[0].shape
        front_shape = fronts[0].shape
        rgb_shape   = rgbs[0].shape
        top_feature_shape = (top_shape[0]//stride, top_shape[1]//stride)
        out_shape=(8,3)


        #-----------------------
        #check data
        if 0:
            fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
            draw_lidar(lidars[0], fig=fig)
            draw_gt_boxes3d(gt_boxes3d[0], fig=fig)
            mlab.show(1)
            cv2.waitKey(1)

    # set anchor boxes
    num_class = 2 #incude background
    anchors, inside_inds =  make_anchors(bases, stride, top_shape[0:2], top_feature_shape[0:2])
    inside_inds = np.arange(0,len(anchors),dtype=np.int32)  #use all  #<todo>
    print ('out_shape=%s'%str(out_shape))
    print ('num_frames=%d'%num_frames)


    #load model ####################################################################################################
    top_anchors     = tf.placeholder(shape=[None, 4], dtype=tf.int32,   name ='anchors'    )
    top_inside_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name ='inside_inds')

    top_images   = tf.placeholder(shape=[None, *top_shape  ], dtype=tf.float32, name='top'  )
    front_images = tf.placeholder(shape=[None, *front_shape], dtype=tf.float32, name='front')
    rgb_images   = tf.placeholder(shape=[None, *rgb_shape  ], dtype=tf.float32, name='rgb'  )
    top_rois     = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='top_rois'   ) #<todo> change to int32???
    front_rois   = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='front_rois' )
    rgb_rois     = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='rgb_rois'   )

    top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores = \
        top_feature_net(top_images, top_anchors, top_inside_inds, num_bases)

    front_features = front_feature_net(front_images)
    rgb_features   = rgb_feature_net(rgb_images)

    fuse_scores, fuse_probs, fuse_deltas = \
        fusion_net(
			( [top_features,     top_rois,     6,6,1./stride],
			  [front_features,   front_rois,   0,0,1./stride],  #disable by 0,0
			  [rgb_features,     rgb_rois,     6,6,1./stride],),
            num_class, out_shape) #<todo>  add non max suppression



    #loss ########################################################################################################
    top_inds     = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_ind'    )
    top_pos_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_pos_ind')
    top_labels   = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_label'  )
    top_targets  = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='top_target' )
    top_cls_loss, top_reg_loss = rpn_loss(top_scores, top_deltas, top_inds, top_pos_inds, top_labels, top_targets)

    fuse_labels  = tf.placeholder(shape=[None            ], dtype=tf.int32,   name='fuse_label' )
    fuse_targets = tf.placeholder(shape=[None, *out_shape], dtype=tf.float32, name='fuse_target')
    fuse_cls_loss, fuse_reg_loss = rcnn_loss(fuse_scores, fuse_deltas, fuse_labels, fuse_targets)


    #solver
    l2 = l2_regulariser(decay=0.0005)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    solver = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    solver_step = solver.minimize(top_cls_loss+top_reg_loss+fuse_cls_loss+0.1*fuse_reg_loss+l2)

    max_iter = 20000
    iter_debug = 8

    # start training here  #########################################################################################
    log.write('epoch     iter    rate   |  top_cls_loss   reg_loss   |  fuse_cls_loss  reg_loss  |  \n')
    log.write('-------------------------------------------------------------------------------------\n')

    num_ratios=len(ratios)
    num_scales=len(scales)
    fig, axs = plt.subplots(num_ratios,num_scales)

    sess = tf.InteractiveSession()
    with sess.as_default():
        sess.run( tf.global_variables_initializer(), { IS_TRAIN_PHASE : True } )
        summary_writer = tf.summary.FileWriter(out_dir+'/tf', sess.graph)
        saver  = tf.train.Saver()
        if initial_model is not None:
            saver.restore(sess, initial_model)



        batch_top_cls_loss =0
        batch_top_reg_loss =0
        batch_fuse_cls_loss=0
        batch_fuse_reg_loss=0
        for iter in range(max_iter):
            epoch=1.0*iter
            rate=0.05


            ## generate train image -------------
            idx = np.random.choice(num_frames)     #*10   #num_frames)  #0
            batch_top_images    = tops[idx].reshape(1,*top_shape)
            batch_front_images  = fronts[idx].reshape(1,*front_shape)
            batch_rgb_images    = rgbs[idx].reshape(1,*rgb_shape)

            batch_gt_labels    = gt_labels[idx]
            batch_gt_boxes3d   = gt_boxes3d[idx]
            batch_gt_top_boxes = box3d_to_top_box(batch_gt_boxes3d)


			## run propsal generation ------------
            fd1={
                top_images:      batch_top_images,
                top_anchors:     anchors,
                top_inside_inds: inside_inds,

                learning_rate:   rate,
                IS_TRAIN_PHASE:  True
            }
            batch_proposals, batch_proposal_scores, batch_top_features = sess.run([proposals, proposal_scores, top_features],fd1)

            ## generate  train rois  ------------
            batch_top_inds, batch_top_pos_inds, batch_top_labels, batch_top_targets  = \
                rpn_target ( anchors, inside_inds, batch_gt_labels,  batch_gt_top_boxes)

            batch_top_rois, batch_fuse_labels, batch_fuse_targets  = \
                 rcnn_target(  batch_proposals, batch_gt_labels, batch_gt_top_boxes, batch_gt_boxes3d )

            batch_rois3d	 = project_to_roi3d    (batch_top_rois)
            batch_front_rois = project_to_front_roi(batch_rois3d  )
            batch_rgb_rois   = project_to_rgb_roi  (batch_rois3d  )


            ##debug gt generation
            if 1 and iter%iter_debug==0:
                top_image = top_imgs[idx]
                rgb       = rgbs[idx]


                img_label  = draw_rcnn_labels (top_image, batch_top_rois, batch_fuse_labels )
                img_target = draw_rcnn_targets(top_image, batch_top_rois, batch_fuse_labels, batch_fuse_targets)
                imshow('img_rcnn_target',img_target)


                img_rgb_rois = draw_boxes(rgb, batch_rgb_rois[:,1:5], color=(255,0,255), thickness=1)
                imshow('img_rgb_rois',img_rgb_rois)

                cv2.waitKey(1)

            ## run classification and regression loss -----------
            fd2={
				**fd1,

                top_images: batch_top_images,
                front_images: batch_front_images,
                rgb_images: batch_rgb_images,

				top_rois:   batch_top_rois,
                front_rois: batch_front_rois,
                rgb_rois:   batch_rgb_rois,

                top_inds:     batch_top_inds,
                top_pos_inds: batch_top_pos_inds,
                top_labels:   batch_top_labels,
                top_targets:  batch_top_targets,

                fuse_labels:  batch_fuse_labels,
                fuse_targets: batch_fuse_targets,
            }
            #_, batch_top_cls_loss, batch_top_reg_loss = sess.run([solver_step, top_cls_loss, top_reg_loss],fd2)


            _, batch_top_cls_loss, batch_top_reg_loss, batch_fuse_cls_loss, batch_fuse_reg_loss = \
               sess.run([solver_step, top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss],fd2)

            log.write('%3.1f   %d   %0.4f   |   %0.5f   %0.5f   |   %0.5f   %0.5f  \n' %\
				(epoch, iter, rate, batch_top_cls_loss, batch_top_reg_loss, batch_fuse_cls_loss, batch_fuse_reg_loss))


            # debug: ------------------------------------

            if iter%iter_debug==0:
                top_image = top_imgs[idx]
                rgb       = rgbs[idx]

                batch_top_probs,  batch_top_deltas  =  sess.run([ top_probs,  top_deltas  ],fd2)
                batch_fuse_probs, batch_fuse_deltas =  sess.run([ fuse_probs, fuse_deltas ],fd2)


                probs, boxes3d = rcnn_nms(batch_fuse_probs, batch_fuse_deltas, batch_rois3d, threshold=0.5)
                cv2.waitKey(1)

                ## show rcnn(fuse) nms
                img_rcnn     = draw_rcnn (top_image, batch_fuse_probs, batch_fuse_deltas, batch_top_rois, batch_rois3d,darker=1)
                img_rcnn_nms1 = draw_rcnn_nms_rgb(rgb, boxes3d, probs)
                img_rcnn_nms2 = draw_rcnn_nms_top(top_image, boxes3d, probs)

                imshow('img_rcnn',img_rcnn)
                imshow('img_rcnn_nms1',img_rcnn_nms1)
                imshow('img_rcnn_nms2',img_rcnn_nms2)
                cv2.waitKey(1)

            # save: ------------------------------------
            if iter%500==0:
                saver.save(sess, out_dir + '/check_points/snap.ckpt',global_step=0)  #iter

## test multi-camera for soccer seqence ###################################################


if __name__ == '__main__':
    run_train()

