import cv2

image_ori = cv2.imread("./../vedio_demo/1.jpg")

video_size = (image_ori.shape[1],image_ori.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter("res.mp4",  fourcc, 30, video_size, True)

for i in range(0, 250):
    frame = cv2.imread("./../vedio_demo/" + str(i) + ".jpg")
    video.write(frame)

