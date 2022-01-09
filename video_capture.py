import cv2 as cv
import numpy

video_path = './data/chessboard.mp4'
save_path = './data/private'
# mm:sec
start_time = '00:43' 
end_time = '00:48'
interval_time = 0.5 # sec


start_seq = [int(i) for i in start_time.split(':')]
start_sec = start_seq[0] * 60 + start_seq[1]
end_seq = [int(i) for i in end_time.split(':')]
end_sec = (end_seq[0]) * 60 + end_seq[1]
assert(start_sec < end_sec)


capture = cv.VideoCapture(video_path)
fps = capture.get(cv.CAP_PROP_FPS)
valid_frame_nu = int(fps * (end_sec - start_sec))
interval_frame_nu = int(fps * interval_time)


capture.set(cv.CAP_PROP_POS_MSEC, start_sec*1e3)

frame_pos = 0
i = 0
while True:
    ret, cap_img = capture.read()
    if not ret or frame_pos > valid_frame_nu:
        break
    if frame_pos % interval_frame_nu == 0:
        img_path = save_path + '/' + str(i) + '.png'
        cv.imwrite(img_path,cap_img)
        i += 1
    frame_pos += 1

print('captured images are saved to \'{}\''.format(save_path))