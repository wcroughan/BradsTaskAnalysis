# Trodes pausing when it loses LEDs is annoying so I'm just writing my own version

import io
import ffmpeg
import numpy as np
# import cv2
import struct
import os
import matplotlib.pyplot as plt
import time

from UtilFunctions import readRawPositionData, processPosData


def getFrameBatch(videoFileName, startFrame, numFrames=100):
    process = (
        ffmpeg
        .input(videoFileName)
        .video
        .filter('select', 'gte(n,{})'.format(startFrame))
        .output('pipe:', format='rawvideo', vframes=numFrames, pix_fmt='rgb24')
        .run_async(pipe_stdin=True, pipe_stdout=True)
    )

    stream = io.BytesIO()
    stream.seek(0)

    outs, errs = process.communicate(input=stream.getvalue())
    return outs


def processRawTrodesVideo(videoFileName, timestampFileName=None, lightOffThreshold=0.1,
                          threshold=50, searchDist=100, showVideo=False, frameBatchSize=1000,
                          maxNumBatches=None, outputFileName=None, batchStart=0,
                          overwriteWithoutConfirm=False):
    probe = ffmpeg.probe(videoFileName)
    video_stream = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    lightOffThreshold *= 255 * width * height
    # print(lightOffThreshold)

    ratXLoc, ratYLoc = 0, 0

    STATE_START_LIGHTSON = 0
    STATE_START_LIGHTSOFF = 1
    STATE_TASK = 2
    STATE_END_LIGHTSON = 3
    currentState = STATE_START_LIGHTSON

    if timestampFileName is None:
        timestampFileName = '.'.join(videoFileName.split('.')[0:-1]) + '.videoTimeStamps'

    with open(timestampFileName, "rb") as tsFile:
        max_iter = 8
        iter = 0
        l = ''
        while l != b'<end settings>\n':
            l = tsFile.readline().lower()
            # print(l)
            iter += 1
            if iter > max_iter:
                raise Exception
        timeStamps = np.fromfile(tsFile, np.uint32)

    if outputFileName is None:
        outputFileName = '.'.join(videoFileName.split('.')[0:-1]) + '.videoPositionTracking'

    if os.path.exists(outputFileName) and not overwriteWithoutConfirm:
        c = input("Output file {} exists, overwrite? (y/N):".format(outputFileName))
        if c != "y":
            return

    outputDataType = np.dtype([('timestamp', np.uint32), ('x1', np.uint16),
                               ('y1', np.uint16), ('x2', np.uint16), ('y2', np.uint16)])

    outputArray = np.zeros((len(timeStamps),), dtype=outputDataType)
    outputIdx = 0
    timestampOffset = batchStart

    quitReq = False
    batchNum = -1
    while True:
        batchNum += 1
        if maxNumBatches is not None and batchNum > maxNumBatches:
            break

        batch = getFrameBatch(videoFileName, batchStart, numFrames=frameBatchSize)
        batchStart += frameBatchSize

        if not batch:
            break

        batchBufferIdx = 0
        batchBufferIdxJump = width * height * 3
        while True:
            frame = batch[batchBufferIdx:batchBufferIdx + batchBufferIdxJump]
            batchBufferIdx += batchBufferIdxJump

            if not frame:
                break

            if showVideo:
                npframe = np.frombuffer(frame, np.uint8).reshape((height, width, 3)).copy()
            else:
                npframe = np.frombuffer(frame, np.uint8).reshape((height, width, 3))

            # print("state = {}".format(currentState))

            # =====================
            # Now we're actually looking at a frame and analyzing it!

            if currentState == STATE_START_LIGHTSON:
                # Just waiting for lights off
                brightness = np.sum(npframe)
                # print("{} (thresh = {})".format(brightness, lightOffThreshold))
                # print("b", brightness)
                # print("t", lightOffThreshold)
                if brightness < lightOffThreshold:
                    print("state start lighton --> start lightoff")
                    currentState = STATE_START_LIGHTSOFF
                    x1 = 0
                    x2 = int(width / 2)
                    y1 = 0
                    y2 = int(height / 2)

            elif currentState == STATE_START_LIGHTSOFF:
                # Now just waiting for the rat
                # Keep checking upper left part of video for anything above threshold
                x1 = 0
                x2 = int(width / 2)
                y1 = 0
                y2 = int(height / 2)
                rchan = npframe[y1:y2, x1:x2, 2].flatten()
                gchan = npframe[y1:y2, x1:x2, 1].flatten()
                if np.count_nonzero(rchan > threshold) > 0 and np.count_nonzero(gchan > threshold):
                    # A light!
                    print("state start lightoff --> task")
                    currentState = STATE_TASK

                    # Where is the light?
                    mgidx = np.argmax(gchan)
                    mgcoords = np.unravel_index(mgidx, (y2, x2))
                    mridx = np.argmax(rchan)
                    mrcoords = np.unravel_index(mridx, (y2, x2))
                    ratXLoc = int((mrcoords[1] + mgcoords[1]) / 2.0)
                    ratYLoc = int((mrcoords[0] + mgcoords[0]) / 2.0)

            elif currentState == STATE_TASK:
                # this includes all the way until the lights go on again.
                x1 = max(0, ratXLoc - searchDist)
                x2 = min(width, ratXLoc + searchDist)
                y1 = max(0, ratYLoc - searchDist)
                y2 = min(height, ratYLoc + searchDist)
                rchan = npframe[y1:y2, x1:x2, 2].flatten()
                gchan = npframe[y1:y2, x1:x2, 1].flatten()

                subHeight = y2 - y1
                subWidth = x2 - x1

                mgidx = np.argmax(gchan)
                mgcoords = np.unravel_index(mgidx, (subHeight, subWidth))
                mridx = np.argmax(rchan)
                mrcoords = np.unravel_index(mridx, (subHeight, subWidth))

                ratXLoc = int((mrcoords[1] + mgcoords[1]) / 2.0) + x1
                ratYLoc = int((mrcoords[0] + mgcoords[0]) / 2.0) + y1

            if showVideo:
                if currentState == STATE_TASK:
                    for i in range(-3, 3):
                        for j in range(-3, 3):
                            npframe[mgcoords[0] + i + y1, mgcoords[1] + j + x1, 0] = 255
                            npframe[mgcoords[0] + i + y1, mgcoords[1] + j + x1, 1] = 255
                            npframe[mgcoords[0] + i + y1, mgcoords[1] + j + x1, 2] = 0
                    for i in range(-3, 3):
                        for j in range(-3, 3):
                            npframe[mrcoords[0] + i + y1, mrcoords[1] + j + x1, 0] = 255
                            npframe[mrcoords[0] + i + y1, mrcoords[1] + j + x1, 1] = 0
                            npframe[mrcoords[0] + i + y1, mrcoords[1] + j + x1, 2] = 255

                if currentState == STATE_START_LIGHTSOFF or currentState == STATE_TASK:
                    npframe[y1:y2, x1, :] = 255
                    npframe[y1:y2, x2-1, :] = 255
                    npframe[y1, x1:x2, :] = 255
                    npframe[y2-1, x1:x2, :] = 255

                cv2.imshow("frame", npframe)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    quitReq = True
                    break

            outputArray[outputIdx]['timestamp'] = timeStamps[outputIdx + timestampOffset]
            outputArray[outputIdx]['x1'] = ratXLoc
            outputArray[outputIdx]['y1'] = ratYLoc
            outputArray[outputIdx]['x2'] = 0
            outputArray[outputIdx]['y2'] = 0
            # print(outputArray[outputIdx])
            outputIdx += 1

        if quitReq:
            break

    print("saving to file {}...".format(outputFileName))
    outputArray = outputArray[0:outputIdx]
    with open(outputFileName, "wb") as outFile:
        header = "<Start settings>\nThis file was created by TrodesCameraExtractor.py\n" + \
            "In the future I'll add in light on/off times here" + \
            "\n<End settings>\n"
        outFile.write(bytes(header, 'utf-8'))
        outputArray.tofile(outFile)

    # process.wait()
    if showVideo:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    videoFileName = "/home/wcroughan/Desktop/20211215_140433.1.h264"

    # t1 = time.perf_counter()
    # processRawTrodesVideo(videoFileName, timestampFileName=None,
    #                       threshold=50, searchDist=150, showVideo=False, frameBatchSize=2000,
    #                       maxNumBatches=None, outputFileName=None, batchStart=4000, overwriteWithoutConfirm=True)
    # t2 = time.perf_counter()
    processRawTrodesVideo(videoFileName, timestampFileName=None,
                          threshold=50, searchDist=150, showVideo=False, frameBatchSize=1000,
                          maxNumBatches=None, outputFileName=None, batchStart=4000, overwriteWithoutConfirm=True)
    # t3 = time.perf_counter()
    # print("batch size 500: ", t2 - t1)
    # print("batch size 1000: ", t3 - t2)
    # batch size 2000 gets killed ... batch size 1000 is twice as fast as 500

    posFileName = "/home/wcroughan/Desktop/20211215_140433.1.videoPositionTracking"
    originalPosFileName = "/home/wcroughan/Desktop/originalTracking.videoPositionTracking"

    position_data = readRawPositionData(posFileName)
    # print(position_data)
    print(position_data.shape)
    xs, ys, ts = processPosData(position_data)
    # plt.plot(xs, ys)
    plt.plot(ts, xs)
    plt.xlim(ts[0], ts[-1])

    position_data = readRawPositionData(originalPosFileName)
    # print(position_data)
    print(position_data.shape)
    xs, ys, ts = processPosData(position_data)
    # plt.plot(xs, ys)
    plt.plot(ts, xs)
    plt.legend(["py", "trodes"])
    plt.show()
