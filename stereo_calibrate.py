import numpy as np
import cv2 as cv
import glob

# user config -------------------------------------------------------
save = True
# display option
display_corner = True 
display_rectify = True
img_folder_path = './data'
# xml, yml, json are supported only
result_path = './result/res.xml'
grid_length = 30 # mm
corner_size = (9,6) # how many corners the chessboard have
img_size = (640, 480)
# user config end ---------------------------------------------------

def stereo_calibrate(images_left, images_right, display=False):
    '''
    @brief calibrate stereo camera
    @param images_left: uri list of left images
    @param images_right: uri list of right images
    @ret: RMS_error, cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right, R, T, E, F
    '''
    # termination criteria: (type, max iter count, accuracy)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    # prepare object points
    # assume the chessboard was located at still plane whose Z_world = 0
    # also assume corners' coordinates are (0,0,0) (0,1,0) (0,2,0) ... (5,0,0) (5,1,0) ...
    objp_template = np.zeros((corner_size[0]*corner_size[1], 3), np.float32)
    objp_template[:,:2] = np.mgrid[0:corner_size[0],0:corner_size[1]].T.reshape(-1,2)

    # get corner points and store
    imgpoints_l = []
    imgpoints_r = []
    objpoints = []

    assert(len(images_left) == len(images_right)) # numbers of images of both side should be the same

    # find corners and store
    for i in range(len(images_left)):
        img_left = cv.imread(images_left[i])
        img_right = cv.imread(images_right[i])
        img_left_grey = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
        img_right_grey = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

        found_l, corners_l = cv.findChessboardCorners(img_left_grey, corner_size, None)
        found_r, corners_r = cv.findChessboardCorners(img_right_grey, corner_size, None)
        
        
        if found_l and found_r:
            corners_refined_l = cv.cornerSubPix(img_left_grey, corners_l, (11,11), (-1,-1), criteria)
            corners_refined_r = cv.cornerSubPix(img_right_grey, corners_r, (11,11), (-1,-1), criteria)
            imgpoints_l.append(corners_refined_l)
            imgpoints_r.append(corners_refined_r)
            objpoints.append(objp_template)

            if display_corner:
                cv.drawChessboardCorners(img_left, corner_size, corners_refined_l, found_l)
                cv.imshow(images_left[i], img_left)
                cv.drawChessboardCorners(img_right, corner_size, corners_refined_r, found_r)
                cv.imshow(images_right[i], img_right)
                cv.waitKey(500)
    if display_corner:
        print('Press any key to continue...')
        cv.waitKey(0)
        cv.destroyAllWindows()

    # initialize proper value to camera matrix
    init_cam_mat_l = cv.initCameraMatrix2D(objpoints, imgpoints_l, img_size)
    init_cam_mat_r = cv.initCameraMatrix2D(objpoints, imgpoints_r, img_size)

    # calibrate
    calibrate_flag = cv.CALIB_FIX_ASPECT_RATIO | cv.CALIB_FIX_FOCAL_LENGTH | \
                    cv.CALIB_USE_INTRINSIC_GUESS | cv.CALIB_SAME_FOCAL_LENGTH
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, 
                    cameraMatrix1=init_cam_mat_l, distCoeffs1=None, 
                    cameraMatrix2=init_cam_mat_r, distCoeffs2=None,
                    imageSize=img_size, flags=calibrate_flag, criteria=criteria)
    print('calibration RMS error: [{}]'.format(retval))

    return retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F


def show_rectify(img_l, img_r, map_x_left, map_y_left, map_x_right, map_y_right, draw_line=True, delay=0, desc=''):
    remap_l = cv.remap(img_l, map_x_left, map_y_left, cv.INTER_LINEAR)
    remap_r = cv.remap(img_r, map_x_right, map_y_right, cv.INTER_LINEAR)
    canvas = cv.hconcat([remap_l, remap_r])
    for i in range(20, canvas.shape[0], 20):
        cv.line(canvas, (0,i), (canvas.shape[1], i), (0, 255, 0), 1)
    cv.imshow('rectify compare '+desc, canvas)
    cv.waitKey(delay)

if __name__ == '__main__':

    images_left = glob.glob(img_folder_path+'/left*.jpg')
    images_right = glob.glob(img_folder_path+'/right*.jpg')

    RMS_error, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
        stereo_calibrate(images_left, images_right, display=display_corner)

    # rectify
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
        cv.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, 
                        img_size, R, T, flags=cv.CALIB_ZERO_DISPARITY) 
    # CALIB_ZERO_DISPARITY: the principal points of each camera have the same pixel coordinates in the rectified views
    

    if display_rectify:
        # precompute pixel map from image to rectified image
        map1_l, map2_l = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, img_size, cv.CV_16SC2)
        map1_r, map2_r = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, img_size, cv.CV_16SC2)

        for i in range(len(images_left)):
            img_left = cv.imread(images_left[i])
            img_right = cv.imread(images_right[i])
            show_rectify(img_left, img_right, map1_l, map2_l, map1_r, map2_r, delay=500, desc=str(i))
        print('Press any key to continue...')
        cv.waitKey(0)
        cv.destroyAllWindows()

    if save:
        fs = cv.FileStorage(result_path, cv.FileStorage_WRITE)
        fs.write('RMS_error', RMS_error)
        fs.write('cameraMatrix1', cameraMatrix1)
        fs.write('distCoeffs1', distCoeffs1)
        fs.write('cameraMatrix1', cameraMatrix1)
        fs.write('distCoeffs1', distCoeffs1)
        fs.write('relative_rotation', R)
        fs.write('relative_transformation', T)
        fs.write('essential_matrix', E)
        fs.write('fundamental_matrix', F)
        fs.release()

    print('results are saved to path: \'{}\''.format(result_path))