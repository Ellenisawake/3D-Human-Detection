from net.utility.draw import *
from net.processing.boxes3d import *

from plyfile import PlyData, PlyElement
from sklearn.cluster import MeanShift, estimate_bandwidth

# run functions --------------------------------------------------------------------------

ROOT_DIR = '../3d_human_detection'

## objs to gt boxes ##
def obj_to_gt_boxes3d(objs):

    num        = len(objs)
    gt_boxes3d = np.zeros((num,8,3),dtype=np.float32)
    gt_labels  = np.zeros((num),    dtype=np.int32)

    for n in range(num):
        obj = objs[n]
        b   = obj.box
        label = 1 #<todo>

        gt_labels [n]=label
        gt_boxes3d[n]=b

    return  gt_boxes3d, gt_labels


## lidar to top ##
def lidar_to_top0(lidar):

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)//TOP_Z_DIVISION)+1
    height  = Yn - Y0
    width   = Xn - X0
    channel = Zn - Z0  + 2

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)

    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)

    ## start to make top  here !!!
    for z in range(Z0,Zn):
        iz = np.where (qzs==z)
        for y in range(Y0,Yn):
            iy  = np.where (qys==y)
            iyz = np.intersect1d(iy, iz)

            for x in range(X0,Xn):
                #print('', end='\r',flush=True)
                #print(z,y,z,flush=True)

                ix = np.where (qxs==x)
                idx = np.intersect1d(ix,iyz)

                if len(idx)>0:
                    yy,xx,zz = -(x-X0),-(y-Y0),z-Z0


                    #height per slice
                    max_height = max(0,np.max(pzs[idx])-TOP_Z_MIN)
                    top[yy,xx,zz]=max_height

                    #intensity
                    max_intensity = np.max(prs[idx])
                    top[yy,xx,Zn]=max_intensity

                    #density
                    count = len(idx)
                    top[yy,xx,Zn+1]+=count

                pass
            pass
        pass
    top[:,:,Zn+1] = np.log(top[:,:,Zn+1]+1)/math.log(64)

    if 1:
        top_image = np.sum(top,axis=2)
        top_image = top_image-np.min(top_image)
        top_image = (top_image/np.max(top_image)*255)
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)


    if 0: #unprocess
        top_image = np.zeros((height,width,3),dtype=np.float32)

        num = len(lidar)
        for n in range(num):
            x,y = qxs[n],qys[n]
            if x>=0 and x <width and y>0 and y<height:
                top_image[y,x,:] += 1

        max_value=np.max(np.log(top_image+0.001))
        top_image = top_image/max_value *255
        top_image=top_image.astype(dtype=np.uint8)


    return top, top_image




## lidar to top ##
def lidar_to_top(lidar):


    idx = np.where (lidar[:,0]>TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,0]<TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar[:,1]>TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,1]<TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar[:,2]>TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,2]<TOP_Z_MAX)
    lidar = lidar[idx]

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]
    qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)


    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)//TOP_Z_DIVISION)+1
    height  = Yn - Y0
    width   = Xn - X0
    channel = Zn - Z0  + 2
    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)

    ## start to make top  here !!!
    for z in range(Z0,Zn):
        iz = np.where (qzs==z)
        for y in range(Y0,Yn):
            iy  = np.where (qys==y)
            iyz = np.intersect1d(iy, iz)

            for x in range(X0,Xn):
                #print('', end='\r',flush=True)
                #print(z,y,z,flush=True)

                ix = np.where (qxs==x)
                idx = np.intersect1d(ix,iyz)

                if len(idx)>0:
                    yy,xx,zz = -(x-X0),-(y-Y0),z-Z0


                    #height per slice
                    max_height = max(0,np.max(pzs[idx])-TOP_Z_MIN)
                    top[yy,xx,zz]=max_height

                    #intensity
                    max_intensity = np.max(prs[idx])
                    top[yy,xx,Zn]=max_intensity

                    #density
                    count = len(idx)
                    top[yy,xx,Zn+1]+=count

                pass
            pass
        pass
    top[:,:,Zn+1] = np.log(top[:,:,Zn+1]+1)/math.log(64)

    if 1:
        top_image = np.sum(top,axis=2)
        top_image = top_image-np.min(top_image)
        top_image = (top_image/np.max(top_image)*255)
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)


    if 0: #unprocess
        top_image = np.zeros((height,width,3),dtype=np.float32)

        num = len(lidar)
        for n in range(num):
            x,y = qxs[n],qys[n]
            if x>=0 and x <width and y>0 and y<height:
                top_image[y,x,:] += 1

        max_value=np.max(np.log(top_image+0.001))
        top_image = top_image/max_value *255
        top_image=top_image.astype(dtype=np.uint8)


    return top, top_image

## drawing ####

def draw_lidar(lidar, is_grid=False, is_top_region=True, fig=None):

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    #prs=lidar[:,3]


    if fig is None: fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))

    # mlab.points3d(
    #     pxs, pys, pzs, prs,
    #     mode='point',  # 'point'  'sphere'
    #     colormap='gnuplot',  #'bone',  #'spectral',  #'copper',
    #     scale_factor=1,
    #     figure=fig)

    mlab.points3d(
        pxs, pys, pzs,
        mode='point',  # 'point'  'sphere'
        #colormap='gnuplot',  #'bone',  #'spectral',  #'copper',
        scale_factor=1,
        figure=fig)


    #draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        for y in np.arange(-50,50,1):
            x1,y1,z1 = -50, y, 0
            x2,y2,z2 =  50, y, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

        for x in np.arange(-50,50,1):
            x1,y1,z1 = x,-50, 0
            x2,y2,z2 = x, 50, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    #draw axis
    if 1:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)
        fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [20., 20., 0.,0.],
            [20.,-20., 0.,0.],
        ],dtype=np.float64)


        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    #draw top_image feature area
    if is_top_region:
        x1 = TOP_X_MIN
        x2 = TOP_X_MAX
        y1 = TOP_Y_MIN
        y2 = TOP_Y_MAX
        mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)



    mlab.orientation_axes()
    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991
    print(mlab.view())



def draw_gt_boxes3d(gt_boxes3d, fig, is_number=False, color=(1,1,1), line_width=2):

    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]

        if is_number:
            mlab.text3d(b[0,0], b[0,1], b[0,2], '%d'%n, scale=(1, 1, 1), color=color, figure=fig)

        for k in range(0,4):

            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991



# soccer-seq #################################################################

# #correction
MATRIX_T = np.array([
    [1, 0, 0,  8],
    [0, 0, 1,  0],
    [0,-1, 0,2.3],
    [0, 0, 0,  1],
])
MATRIX_Tinv = np.linalg.inv(MATRIX_T)

def T_xyz(x,y,z):
    return x+8, z, -y+2.3  #swap y to z



def read_cameras(camera_file):

    lines = []
    with open(camera_file) as file:
        for line in file:
            line = line.strip() #or someother preprocessing
            lines.append(line)

    num_cameras = int(lines[16])
    print('num_cameras=%d'%num_cameras)

    cameras=[]
    for n in range(num_cameras):
        N = 18+n*14
        cam = type('', (), {})()
        cam.img_file=lines[N+0]

        focal  = float(lines[N+2])
        center = [float(x) for x in lines[N+3].split()]

        c  = [float(x) for x in lines[N+ 5].split()]
        t  = [float(x) for x in lines[N+ 4].split()]
        r0 = [float(x) for x in lines[N+ 8].split()]
        r1 = [float(x) for x in lines[N+ 9].split()]
        r2 = [float(x) for x in lines[N+10].split()]

        cam.K=np.array(
            [[focal,0,center[0]],
             [0,focal,center[1]],
             [0,0,1]]
        )
        cam.M=np.array(
            [[r0[0],r0[1],r0[2],t[0]],
             [r1[0],r1[1],r1[2],t[1]],
             [r2[0],r2[1],r2[2],t[2]],
             [0,0,0,1]]
        )
        cam.c=np.array(c)
        cam.focal=focal
        cam.center=center

        ##correction
        cam.M = np.matmul(cam.M,MATRIX_Tinv)
        cam.c = T_xyz(cam.c[0],cam.c[1],cam.c[2])

        cameras.append(cam)

    return cameras

def read_points(point_file):
    plydata = PlyData.read(point_file)
    num_points=plydata['vertex'].count
    points=np.zeros((num_points,3),dtype=np.float32)  #ignore rgb
    for n in range(num_points):
        x,y,z = plydata['vertex'][n][0], plydata['vertex'][n][1], plydata['vertex'][n][2]
        points[n]= T_xyz(x,y,z)

    return points

def solve_intersect_point(direction_vectors, centers):
    ##  https://jp.mathworks.com/matlabcentral/fileexchange/37192-intersection-point-of-lines-in-3d-space
    ##  https://github.com/sbenodiz/Shape_Recover/blob/master/Computer%20Vision/Global_SecondOrder/lineIntersect3D.m
    ##  http://math.stackexchange.com/questions/1911106/intersection-point-of-multiple-3d-lines/1911188
    ##  http://math.stackexchange.com/questions/61719/finding-the-intersection-point-of-many-lines-in-3d-point-closest-to-all-lines
    ##  http://www.staff.city.ac.uk/~sbbh653/publications/opray.pdf

    unit_vectors = direction_vectors/np.linalg.norm(direction_vectors,axis=1,keepdims=True)
    nx = unit_vectors[:,0]
    ny = unit_vectors[:,1]
    nz = unit_vectors[:,2]

    SXX = np.sum(nx*nx-1)
    SYY = np.sum(ny*ny-1)
    SZZ = np.sum(nz*nz-1)
    SXY = np.sum(nx*ny)
    SXZ = np.sum(nx*nz)
    SYZ = np.sum(ny*nz)
    S   = np.array([[SXX, SXY, SXZ],[SXY, SYY, SYZ],[SXZ, SYZ, SZZ]])

    CX  = sum(centers[:,0]*(nx*nx-1)  + centers[:,1]*(nx*ny)   + centers[:,2]*(nx*nz))
    CY  = sum(centers[:,0]*(nx*ny)    + centers[:,1]*(ny*ny-1) + centers[:,2]*(ny*nz))
    CZ  = sum(centers[:,0]*(nx*nz)    + centers[:,1]*(ny*nz)   + centers[:,2]*(nz*nz-1))
    C   = np.array([[CX],[CY],[CZ]])

    # Sx = C
    Sinv = np.linalg.inv(S)
    intersect_point = np.matmul(Sinv,C)

    return intersect_point


def soccer_run_00():
    camera_file = ROOT_DIR + '/data/cameras_v2.txt'
    cameras = read_cameras(camera_file)

    point_file = ROOT_DIR + '/data/0000.ply'
    points = read_points(point_file)

    boxes3d = np.load(ROOT_DIR + '/data/gt_boxes3d_00000.npy')
    num_objects=len(boxes3d)

    # --- draw -----
    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
    pxs=points[:,0]
    pys=points[:,1]
    pzs=points[:,2]

    mlab.points3d(
        pxs, pys, pzs,
        mode='point', 
        color=(1, 1, 1),
        scale_factor=1,
        figure=fig)
    draw_gt_boxes3d(boxes3d,fig, color=(1,0,0), line_width=1)

    #show camera
    num_cameras =  len(cameras)
    for n in range(0,num_cameras,1):
        cam = cameras[n]
        mlab.points3d(
            cam.c[0], cam.c[1], cam.c[2],
            mode='sphere',  # 'point'  'sphere'
            color=(0, 1, 0),
            scale_factor=0.2,
            figure=fig)


    mlab.show()
    #mlab.show(1)


    #---
    projections = np.zeros((num_cameras,num_objects,8,2))
    for n in range(0,num_cameras,1):
        cam = cameras[n]
        Mt = np.transpose(cam.M)
        Kt = np.transpose(cam.K)
        projection = box3d_to_rgb_projections(boxes3d, Mt=Mt, Kt=Kt)
        projections[n] = projection

        rgb = cv2.imread('/home/ellen/F/PROJECT/222222/soccer_nagoya/cmvs/00/visualize/' + cam.img_file,1)
        img_rgb = draw_rgb_projections(rgb,  projection, color=(255,255,255), thickness=1)

        imshow('img_rgb%d'%n,img_rgb,resize=0.3)
        cv2.waitKey(1)

    ## do for each point
    for i in range(8):

        centers = np.zeros((num_cameras,3), dtype=np.float32)
        direction_vectors = np.zeros((num_cameras,3), dtype=np.float32)

        for n in range(0,num_cameras,1):
            u,v = projections[n][0,i]

            cam          = cameras[n]
            rotation     = cam.M[0:3,0:3]
            rotationinv  = np.linalg.inv(rotation)
            rotationinvt = rotationinv.transpose()
            Kinv  = np.linalg.inv(cam.K)
            Kinvt = Kinv.transpose()

            q  = np.array([u,v,1])
            q  = np.matmul(q,Kinvt)
            Q  = np.matmul(q,rotationinvt)

            centers[n] = cam.c
            direction_vectors[n] = Q

            pass

        ###  solving least sqaure to find intersection 3d point ###
        p_intersect = solve_intersect_point(direction_vectors, centers)
        print (boxes3d)
        print (p_intersect)

        mlab.points3d(
            p_intersect[0], p_intersect[1], p_intersect[2],
            mode='sphere',  # 'point'  'sphere'
            color=(1, 1, 0),
            scale_factor=0.2,
            figure=fig)

    mlab.show()
    pass

# click point and store
def soccer_run_01():
    from click_points import click_xy

    camera_file = ROOT_DIR + '/data/cameras_v2.txt'
    cameras = read_cameras(camera_file)

    num_cameras=len(cameras)
    for n in range(0,num_cameras,1):
        cam = cameras[n]
        rgb = cv2.imread(ROOT_DIR + '/data/rgb/' + cam.img_file,1)

        m = click_xy(rgb)
        print ('n=%d'%n)
        print (m)
        print ('')

    pass


if __name__ == '__main__':
    soccer_run_00()
    exit(0)

