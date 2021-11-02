import sys
from cv2 import cv2
import os
import time
import datetime
import numpy as np

SHOW_VID = False
SAVE_VID = True
SAVE_ORIGINAL_VID = True
SAVE_IMGS = True

data_dirs = ["/media/WDC6/B12/", "/path/to/your/data/folder"]
data_dir = ""
for dd in data_dirs:
    if os.path.exists(dd):
        data_dir = dd
        break
if data_dir == "":
    print("Couldn't find any of the folders listed in data_dirs. Add the folder on your machine to the list.")
    exit()

videoFileName = "2021-09-16_15-21-25.mp4"
input_video_path = os.path.join(data_dir, videoFileName)
output_video_path = os.path.join(data_dir, "conversion", "outvid.avi")
output_original_video_path = os.path.join(data_dir, "conversion", "outvid_original.avi")
output_image_path = os.path.join(data_dir, "conversion")


class MyVideoAnalyzer:
    def __init__(self):
        self.left_box1 = (575, 420)
        self.left_box2 = (1010, 700)

        self.right_box1 = (1010, 420)
        self.right_box2 = (1420, 700)

        self.run_pfx = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    def process_frame(self, frame, frame_index):
        retframe = frame.copy()

        if SAVE_IMGS and frame_index == 0:
            tmp0 = frame.copy()
            cv2.rectangle(tmp0, self.left_box1, self.left_box2, (255, 255, 255))
            cv2.rectangle(tmp0, self.right_box1, self.right_box2, (255, 255, 255))
            cv2.imwrite(os.path.join(output_image_path, "processing_0.png"), tmp0)

            tmp1 = frame.copy()
            mask = np.array([[865, 420], [1010, 620], [1010, 420]])
            cv2.polylines(tmp1, [mask], True, (255, 255, 255))
            mask = np.array([[760, 700], [1010, 640], [1010, 700]])
            cv2.polylines(tmp1, [mask], True, (255, 255, 255))
            mask = np.array([[1340, 420], [1420, 620], [1420, 420]])
            cv2.polylines(tmp1, [mask], True, (255, 255, 255))
            mask = np.array([[1010, 700], [1010, 685], [1420, 650], [1420, 700]])
            cv2.polylines(tmp1, [mask], True, (255, 255, 255))
            mask = np.array([[1010, 600], [1140, 700], [1035, 700], [1010, 645]])
            cv2.polylines(tmp1, [mask], True, (255, 255, 255))
            cv2.imwrite(os.path.join(output_image_path, "processing_1.png"), tmp1)

            tmp2 = frame.copy()
            mask = np.array([[865, 420], [1010, 620], [1010, 420]])
            cv2.fillPoly(tmp2, pts=[mask], color=(0, 0, 0))
            mask = np.array([[760, 700], [1010, 640], [1010, 700]])
            cv2.fillPoly(tmp2, pts=[mask], color=(0, 0, 0))
            mask = np.array([[1340, 420], [1420, 620], [1420, 420]])
            cv2.fillPoly(tmp2, pts=[mask], color=(0, 0, 0))
            mask = np.array([[1010, 700], [1010, 685], [1420, 650], [1420, 700]])
            cv2.fillPoly(tmp2, pts=[mask], color=(0, 0, 0))
            mask = np.array([[1010, 600], [1140, 700], [1035, 700], [1010, 645]])
            cv2.fillPoly(tmp2, pts=[mask], color=(0, 0, 0))
            cv2.imwrite(os.path.join(output_image_path, "processing_2.png"), tmp2)

        # apply mask for angled corners
        mask = np.array([[865, 420], [1010, 620], [1010, 420]])
        cv2.fillPoly(retframe, pts=[mask], color=(0, 0, 0))
        # cv2.polylines(retframe, [mask], True, (255, 255, 255))
        mask = np.array([[760, 700], [1010, 640], [1010, 700]])
        cv2.fillPoly(retframe, pts=[mask], color=(0, 0, 0))
        # cv2.polylines(retframe, [mask], True, (255, 255, 255))
        mask = np.array([[1340, 420], [1420, 620], [1420, 420]])
        cv2.fillPoly(retframe, pts=[mask], color=(0, 0, 0))
        # cv2.polylines(retframe, [mask], True, (255, 255, 255))
        mask = np.array([[1010, 700], [1010, 685], [1420, 650], [1420, 700]])
        cv2.fillPoly(retframe, pts=[mask], color=(0, 0, 0))
        # cv2.polylines(retframe, [mask], True, (255, 255, 255))
        mask = np.array([[1010, 600], [1140, 700], [1035, 700], [1010, 645]])
        cv2.fillPoly(retframe, pts=[mask], color=(0, 0, 0))
        # cv2.polylines(retframe, [mask], True, (255, 255, 255))

        # grab both cameras
        img_l = frame[self.left_box1[1]:self.left_box2[1],
                      self.left_box1[0]:self.left_box2[0], 0]
        img_r = frame[self.right_box1[1]:self.right_box2[1],
                      self.right_box1[0]:self.right_box2[0], 0]
        # (y, x). Grayscale so can just take one channel

        thresh = 120
        _, mask = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)

        if SAVE_IMGS and frame_index == 0:
            cv2.imwrite(os.path.join(output_image_path, "processing_3.png"), mask)

        kernel = np.ones((3, 3))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        if SAVE_IMGS and frame_index == 0:
            cv2.imwrite(os.path.join(output_image_path, "processing_4.png"), mask)

        retframe = cv2.bitwise_and(retframe, mask)

        if SAVE_IMGS and frame_index == 0:
            cv2.imwrite(os.path.join(output_image_path, "processing_5.png"), retframe)

        # # Find brightest spots
        # maxidxl = np.argmax(img_l)
        # lcircley, lcirclex = np.unravel_index(maxidxl, img_l.shape)
        # lcirclex += self.left_box1[0]
        # lcircley += self.left_box1[1]
        # maxidxr = np.argmax(img_r)
        # rcircley, rcirclex = np.unravel_index(maxidxr, img_r.shape)
        # rcirclex += self.right_box1[0]
        # rcircley += self.right_box1[1]

        # draw both boxes in output
        cv2.rectangle(retframe, self.left_box1, self.left_box2, (255, 255, 255))
        cv2.rectangle(retframe, self.right_box1, self.right_box2, (255, 255, 255))

        # # Add output section to output image
        # cv2.circle(retframe, (lcirclex, lcircley), 3, (0, 0, 255))
        # cv2.circle(retframe, (rcirclex, rcircley), 3, (0, 0, 255))

        return retframe

    def process_video(self, filename, num_frames=None, startTime=None, endTime=None):
        vid = cv2.VideoCapture(filename)
        if not vid.isOpened():
            print("Couldn't open video {}".format(filename))
            return 1

        print("input video is {}x{}".format(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        if SAVE_VID:
            w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fr = vid.get(cv2.CAP_PROP_FPS)
            writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(
                'M', 'J', 'P', 'G'), fr, (w, h))

        if SAVE_ORIGINAL_VID:
            w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fr = vid.get(cv2.CAP_PROP_FPS)
            original_writer = cv2.VideoWriter(output_original_video_path, cv2.VideoWriter_fourcc(
                'M', 'J', 'P', 'G'), fr, (w, h))

        if startTime is not None:
            vid.set(cv2.CAP_PROP_POS_MSEC, startTime)

        framei = 0
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
            if num_frames is not None and framei >= num_frames:
                break

            if endTime is not None and vid.get(cv2.CAP_PROP_POS_MSEC) >= endTime:
                break

            if SAVE_ORIGINAL_VID:
                original_writer.write(frame)

            outframe = self.process_frame(frame, framei)
            if SAVE_VID:
                writer.write(outframe)
                if not SHOW_VID:
                    if endTime is not None:
                        print("{} => {} ({})".format(int(vid.get(cv2.CAP_PROP_POS_MSEC)),
                              int(endTime), int(endTime-vid.get(cv2.CAP_PROP_POS_MSEC)) / 1000))
                    else:
                        print("frame {}".format(framei))

            if SAVE_IMGS and framei == 0:
                cv2.imwrite(os.path.join(output_image_path,
                            "frame_{}.png".format(framei)), outframe)

            if SHOW_VID:
                cv2.imshow('frame', outframe)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            framei += 1

        if SAVE_IMGS:
            cv2.imwrite(os.path.join(output_image_path,
                        "frame_end.png"), outframe)
        if SAVE_VID:
            writer.release()
        if SAVE_ORIGINAL_VID:
            original_writer.release()
        vid.release()
        if SHOW_VID:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    mva = MyVideoAnalyzer()
    # mva.process_video(input_video_path, num_frames=350)
    mva.process_video(input_video_path, num_frames=5, startTime=1194019)
    # mva.process_video(input_video_path, endTime=1409992, startTime=1094019)
