import sys
from cv2 import cv2
import os
import time
import datetime
import numpy as np
import glob
from BTData import *
from queue import Queue
from threading import Thread
from joblib import Parallel, delayed

animal_name = 'B11'

all_well_names = np.array([i + 1 for i in range(48) if not i % 8 in [0, 7]])
well_name_to_idx = np.empty((np.max(all_well_names)+1))
well_name_to_idx[:] = np.nan
for widx, wname in enumerate(all_well_names):
    well_name_to_idx[wname] = widx

if animal_name == "B12_highthresh":
    data_filename = "/media/WDC7/B12/processed_data/B12_highthresh_bradtask.dat"
    output_dir = "/media/WDC6/B12/conversion"
    training_img_dir = "/media/WDC6/B12/conversion"
    out_filename = "B12_conversion_blur.dat"
    video_dir = "/media/WDC6/B12"
elif animal_name == "B11":
    output_dir = "/media/WDC6/B11/conversion"
    training_img_dir = "/media/WDC6/B12/conversion"
    out_filename = "B11_converted.dat"
    video_dir = "/media/WDC6/B11"
else:
    raise Exception("unknonwn dataset")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

makefigs = np.zeros((100,))
makefigs[4] = 1

alldata = BTData()
if makefigs[3]:
    alldata.loadFromFile(os.path.join(output_dir, out_filename))
elif makefigs[4]:
    pass
else:
    alldata.loadFromFile(data_filename)
all_sessions = alldata.getSessions()


if makefigs[0]:
    # Result: cameras stable up through 10/1
    # on 10/4 moved, 10/5 and onward both stable again in a different spot
    # 10/4 and onward is stable just in left cam, right cam isn't stable between 10/4 and 5
    ww = None
    hh = None

    for si, sesh in enumerate(all_sessions):
        t = sesh.sniff_pre_trial_light_off - 500
        fn = sesh.sniffTimesFile[0:-4].split('/')[-1]
        fn = os.path.join(video_dir, fn)
        vid = cv2.VideoCapture(fn)
        if not vid.isOpened():
            print("Couldn't open video {}".format(fn))
            exit()

        vid.set(cv2.CAP_PROP_POS_MSEC, t)

        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if ww is None:
            ww = w
            hh = h
        else:
            if ww != w or hh != h:
                print("{}: vid size change from {} {} to {} {}".format(sesh.date_str, ww, hh, w, h))
                exit()

        ret, frame = vid.read()
        if not ret:
            print("Couldn't get frame :(")
            exit()

        cv2.imwrite(os.path.join(output_dir, "{}_{}.png".format(sesh.date_str, si)), frame)
        vid.release()


VID_FRAME_SZ_1 = 1420 - 575
VID_FRAME_SZ_2 = 700 - 420


def isolateCableInImg(img):
    retframe = img.copy()

    left_box1 = (575, 420)
    # left_box2 = (1010, 700)
    # right_box1 = (1010, 420)
    right_box2 = (1420, 700)

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

    thresh = 120
    _, mask = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    retframe = cv2.bitwise_and(retframe, mask)

    retframe = retframe[left_box1[1]:right_box2[1],
                        left_box1[0]:right_box2[0], 0]

    return retframe


def avgAndSave(snifages, filename):
    print("Saving {}, {} frames".format(filename, np.count_nonzero(
        np.logical_not(np.isnan(snifages[0, 0, :])))))
    outimg = np.nanmean(snifages, axis=2)
    cv2.imwrite(os.path.join(output_dir, filename), outimg)


if makefigs[1]:
    for wi in range(len(all_well_names)):
        print("onto well {}".format(all_well_names[wi]))

        snifages = np.empty((VID_FRAME_SZ_2, VID_FRAME_SZ_1, 100))
        snifages[:] = np.nan
        idx = 0

        for si, sesh in enumerate(all_sessions):
            if sesh.date_str == "20211004":
                avgAndSave(snifages, "pre_{}.png".format(all_well_names[wi]))
                snifages = np.empty((VID_FRAME_SZ_2, VID_FRAME_SZ_1, 100))
                snifages[:] = np.nan
                idx = 0
                continue

            print("session {}".format(sesh.date_str))

            fn = sesh.sniffTimesFile[0:-4].split('/')[-1]
            fn = os.path.join(video_dir, fn)
            vid = cv2.VideoCapture(fn)
            if not vid.isOpened():
                print("Couldn't open video {}".format(fn))
                exit()

            ents = sesh.well_sniff_times_entry[wi]
            exts = sesh.well_sniff_times_exit[wi]
            for ei in range(len(ents)):
                t1 = ents[ei]
                t2 = exts[ei]
                print("({} - {})".format(t1, t2))
                vid.set(cv2.CAP_PROP_POS_MSEC, t1)
                while vid.get(cv2.CAP_PROP_POS_MSEC) < t2:
                    ret, frame = vid.read()
                    if not ret:
                        print("Couldn't get frame :(")
                        exit()

                    pframe = isolateCableInImg(frame)
                    if snifages.shape[2] == idx:
                        a = np.empty((VID_FRAME_SZ_2, VID_FRAME_SZ_1, 100))
                        a[:] = np.nan
                        snifages = np.append(snifages, a, axis=2)
                    snifages[:, :, idx] = pframe
                    idx += 1

            vid.release()

        avgAndSave(snifages, "post_{}.png".format(all_well_names[wi]))


def loadClassificationImages(prefix):
    gl = os.path.join(training_img_dir, prefix + "_*.png")
    imlist = glob.glob(gl)
    assert len(imlist) == len(all_well_names)
    ret = np.empty((VID_FRAME_SZ_2, VID_FRAME_SZ_1, len(imlist)))
    for i, im in enumerate(imlist):
        image = cv2.imread(im)
        blr = cv2.GaussianBlur(image, (21, 21), 10)
        wn = int(im.split('_')[-1].split('.')[0])
        idx = int(well_name_to_idx[wn])
        ret[:, :, idx] = blr[:, :, 0]

    # remember to normalize maybe?! Try both??
    ret = np.divide(ret, np.reshape(np.sum(ret, axis=(0, 1)), (1, 1, -1)))

    # Also probs want to smooth
    return ret


class VideoParser:
    def __init__(self, path, t1, t2, queuesize=64):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.Q = Queue(maxsize=queuesize)

        self.t2 = t2
        self.stream.set(cv2.CAP_PROP_POS_MSEC, t1)

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                break

            if not self.Q.full():
                t = self.stream.get(cv2.CAP_PROP_POS_MSEC)
                if t >= self.t2:
                    self.stop()
                    break

                ret, frame = self.stream.read()

                if not ret:
                    self.stop()
                    break

                self.Q.put((frame, t))

        self.stream.release()

    def read(self):
        return self.Q.get()

    def hasFrame(self):
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True

    def isStopped(self):
        return self.stopped

    def getWidth(self):
        return int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))

    def getHeight(self):
        return int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def getFR(self):
        return self.stream.get(cv2.CAP_PROP_FPS)


def classifyVideo(t1, t2, videoFileName, date_str, outvidName):
    if date_str == "20211004":
        return (None, None)
    elif date_str > "20211004":
        prefix = "post"
    else:
        prefix = "pre"
    avg_images = loadClassificationImages(prefix)
    prod_out = np.empty_like(avg_images)
    addn_out = np.empty((len(all_well_names)))

    post = np.empty((100, len(all_well_names)))
    post[:] = np.nan
    t = np.empty((100,))
    t[:] = np.nan
    idx = 0

    vparse = VideoParser(videoFileName, t1, t2)
    vparse.start()

    w = vparse.getWidth()
    h = vparse.getHeight()
    fr = vparse.getFR()
    writer = cv2.VideoWriter(os.path.join(output_dir, outvidName + "_outvid.avi"), cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), fr, (w, h))

    rt1 = time.time()

    while not vparse.isStopped():
        if not vparse.hasFrame():
            time.sleep(0.1)
            continue

        frame, vparseT = vparse.read()
        pframe = isolateCableInImg(frame)

        # get overlap
        np.multiply(np.reshape(pframe, (VID_FRAME_SZ_2, VID_FRAME_SZ_1, 1)),
                    avg_images, out=prod_out)
        np.sum(prod_out, axis=(0, 1), out=addn_out)

        if post.shape[0] == idx:
            rt2 = time.time()
            rate = (t[-1] - t1) / (rt2 - rt1)
            pct = (t[-1] - t1) / (t2 - t1) * 100
            print("{}% ({} msec/sec)".format(pct, rate))
            a = np.empty((100, len(all_well_names)))
            a[:] = np.nan
            post = np.append(post, a, axis=0)
            a = np.empty((100, ))
            a[:] = np.nan
            t = np.append(t, a)

        post[idx, :] = addn_out
        t[idx] = vparseT

        cv2.putText(frame, str(all_well_names[np.argmax(post[idx, :])]),
                    (1020, 520), cv2.FONT_HERSHEY_COMPLEX, 4, (0, 0, 255))
        writer.write(frame)

        idx += 1

    writer.release()
    return (all_well_names[np.argmax(post, axis=1)], t)


def classifyVideoForSesh(sesh):
    t1 = sesh.sniff_probe_start
    t2 = sesh.sniff_probe_stop

    fn = sesh.sniffTimesFile[0:-4].split('/')[-1]
    fn = os.path.join(video_dir, fn)

    return classifyVideo(t1, t2, fn, sesh.date_str, sesh.name)


def classifyVideoForRgs(rgsfilename):
    with open(rgsfilename, "r") as rf:
        l = rf.readline()
        print(rgsfilename)
        l = rf.readline().split(',')
        t1 = int(l[0])
        t2 = int(l[1])
        fn = rgsfilename[0:-4].split('/')[-1]
        vfn = os.path.join(video_dir, fn)
        date_str = fn[0:4] + fn[5:7] + fn[8:9]
        name = fn[0:-4]

        return classifyVideo(t1, t2, vfn, date_str, name)


if makefigs[2]:
    classResults = Parallel(n_jobs=6)(delayed(classifyVideoForSesh)(s) for s in all_sessions)
    for si, sesh in enumerate(all_sessions):
        sesh.sniffClassificationNearestWell = classResults[si][0]
        sesh.sniffClassificationT = classResults[si][1]

    alldata.saveToFile(os.path.join(output_dir, out_filename))

if makefigs[3]:
    print("STarting")
    for si, sesh in enumerate(all_sessions):
        print(si)
        if si > 0:
            break
        print(sesh.name)

        t1 = sesh.sniff_probe_start
        t2 = sesh.sniff_probe_stop

        fn = sesh.sniffTimesFile[0:-4].split('/')[-1]
        fn = os.path.join(video_dir, fn)
        vid = cv2.VideoCapture(fn)
        if not vid.isOpened():
            print("Couldn't open video {}".format(fn))
            exit()

        vid.set(cv2.CAP_PROP_POS_MSEC, t1)
        fi = 0

        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fr = vid.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(os.path.join(output_dir, sesh.name + "_outvid.avi"), cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), fr, (w, h))

        while vid.get(cv2.CAP_PROP_POS_MSEC) < t2:
            ret, frame = vid.read()
            if not ret:
                print("Couldn't get frame :(")
                exit()

            if fi < len(sesh.sniffClassificationNearestWell):
                cv2.putText(frame, str(
                    sesh.sniffClassificationNearestWell[fi]), (1020, 520), cv2.FONT_HERSHEY_COMPLEX, 4, (0, 0, 255))
                writer.write(frame)

            fi += 1
            if fi % 10 == 0:
                print("{} %".format((vid.get(cv2.CAP_PROP_POS_MSEC) - t1) / (t2 - t1) * 100))

        if fi > len(sesh.sniffClassificationNearestWell):
            print("Warning, video num frames mismatch by {} to {}".format(
                fi, len(sesh.sniffClassificationNearestWell)))

        writer.release()
        vid.release()

if makefigs[4]:
    gl = os.path.join(video_dir, "*.rgs")
    vidlist = glob.glob(gl)
    classResults = Parallel(n_jobs=6)(delayed(classifyVideoForRgs)(v) for v in vidlist)
    # classResults = []
    # for v in vidlist:
    # classResults.append(classifyVideoForRgs(v))
    np.savez(os.path.join(output_dir, "allClass"), np.array(classResults))
