# Trodes pausing when it loses LEDs is annoying so I'm just writing my own version

import io
import ffmpeg
import numpy as np
import os
import glob

if __name__ == "__main__":
    import cv2

from UtilFunctions import getInfoForAnimal, getTrodesVideoFile, getUSBVideoFile
from consts import TRODES_SAMPLING_RATE


def getFrameBatch(videoFileName, startFrame, numFrames=None, frameStride=1):
    if numFrames is None:
        process = (
            ffmpeg
            .input(videoFileName)
            .video
            .filter('select', 'gte(n,{})*not(mod(n,{}))'.format(startFrame, frameStride))
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', vsync="vfr")
            .global_args('-loglevel', 'error')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=True)
        )
    else:
        process = (
            ffmpeg
            .input(videoFileName)
            .video
            .filter('select', 'gte(n,{})*not(mod(n,{}))'.format(startFrame, frameStride))
            .output('pipe:', format='rawvideo', vframes=numFrames, pix_fmt='rgb24', vsync="vfr")
            .global_args('-loglevel', 'error')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=True)
        )

    stream = io.BytesIO()
    stream.seek(0)

    outs, errs = process.communicate(input=stream.getvalue())
    return outs


def getSingleFrame(videoFileName, frameIdx):
    return getFrameBatch(videoFileName, frameIdx, numFrames=1)


def getTrodesLightTimes(videoFileName, timestampFileName=None, initialSkipAmt=8, showVideo=False, outputFileName=None,
                        ignoreFirstSeconds=None):
    if outputFileName is None:
        outputFileName = '.'.join(videoFileName.split('.')[0:-1]) + '.justLights'
    initialFrameJump = np.power(2, initialSkipAmt)

    if timestampFileName is None:
        timestampFileName = '.'.join(videoFileName.split('.')[0:-1]) + '.videoTimeStamps'

    with open(timestampFileName, "rb") as tsFile:
        max_iter = 8
        iter = 0
        line = ''
        while line != b'<end settings>\n':
            line = tsFile.readline().lower()
            # print(line)
            iter += 1
            if iter > max_iter:
                raise Exception
        timeStamps = np.fromfile(tsFile, np.uint32)

    if ignoreFirstSeconds is None:
        ignoreFirstFrames = None
        # print("nothing to ignore at start of video")
    else:
        firstFrameTs = TRODES_SAMPLING_RATE * ignoreFirstSeconds + timeStamps[0]
        ignoreFirstFrames = np.searchsorted(timeStamps, firstFrameTs)
        # print(f"ignoreFirstSeconds={ignoreFirstSeconds}, ignoreFirstFrames={ignoreFirstFrames}")

    # Step one: find left and right bound on light off and on time
    loffF1, loffF2, lonF1, lonF2 = findLightTimesLinearTrodes(
        videoFileName, 0, None, initialFrameJump, showVideo=showVideo, ignoreFirstFrames=ignoreFirstFrames)
    loffF1 *= initialFrameJump
    loffF2 *= initialFrameJump
    lonF1 *= initialFrameJump
    lonF2 *= initialFrameJump

    print("coarse frames ranges:", loffF1, loffF2, lonF1, lonF2)

    # Step two: for each of those intervals, get the video in that range and binary search that ish
    _, lightsOffFrame, _, _ = findLightTimesLinearTrodes(
        videoFileName, loffF1, loffF2, 1, showVideo=showVideo, findLightOn=False)
    lightsOffFrame += loffF1
    _, _, _, lightsOnFrame = findLightTimesLinearTrodes(
        videoFileName, lonF1, lonF2, 1, showVideo=showVideo, findLightOff=False)
    lightsOnFrame += lonF1

    lightsOffTimestamp = timeStamps[lightsOffFrame]
    lightsOnTimestamp = timeStamps[lightsOnFrame]

    outputArr = np.array([lightsOffTimestamp, lightsOnTimestamp])
    outputArr.tofile(outputFileName, sep=",")

    return lightsOffTimestamp, lightsOnTimestamp


def processRawTrodesVideo(videoFileName, timestampFileName=None, lightOffThreshold=0.1,
                          threshold=90, searchDist=230, showVideo=False, frameBatchSize=1000,
                          maxNumBatches=None, outputFileName=None, batchStart=0,
                          overwriteMode="ask", lightOnThreshold=0.2):
    if outputFileName is None:
        outputFileName = '.'.join(videoFileName.split('.')[0:-1]) + '.videoPositionTracking'

    if os.path.exists(outputFileName):
        if overwriteMode == "never":
            return outputFileName

        elif overwriteMode == "rename":
            outputFileName += ".tce"

        elif overwriteMode == "ask":
            c = input("Output file {} exists, overwrite? (y/N/l):".format(outputFileName))
            if c != "y":
                return outputFileName

        elif overwriteMode != "always":
            print("Unknown overwrite mode {}".format(overwriteMode))
            return outputFileName

    if "20211208_180352" in videoFileName:
        # This session has lights on -> off -> on -> off at the start
        minLightTimeStamp = (3600 * 3 + 60 * 51 + 10) * TRODES_SAMPLING_RATE
    if "20220306_180235" in videoFileName:
        # This session has lights on -> off -> on -> off at the start
        minLightTimeStamp = (3600 * 0 + 60 * 1 + 39) * TRODES_SAMPLING_RATE
    elif "20220201_181927" in videoFileName:
        # lights off -> on -> off at start
        minLightTimeStamp = (3600 * 2 + 60 * 24 + 37) * TRODES_SAMPLING_RATE
    else:
        minLightTimeStamp = 0

    probe = ffmpeg.probe(videoFileName)
    video_stream = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    lightOffThreshold *= 255 * width * height
    lightOnThreshold *= 255 * width * height
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
        line = ''
        while line != b'<end settings>\n':
            line = tsFile.readline().lower()
            # print(line)
            iter += 1
            if iter > max_iter:
                raise Exception
        timeStamps = np.fromfile(tsFile, np.uint32)

    outputDataType = np.dtype([('timestamp', np.uint32), ('x1', np.uint16),
                               ('y1', np.uint16), ('x2', np.uint16), ('y2', np.uint16)])

    outputArray = np.zeros((len(timeStamps),), dtype=outputDataType)
    outputIdx = 0
    videoFrame = batchStart
    # timestampOffset = batchStart

    lightsOnFrame = None
    lightsOffFrame = None

    quitReq = False
    batchNum = -1
    while True:
        batchNum += 1
        if maxNumBatches is not None and batchNum > maxNumBatches:
            break

        print("getting batch at", batchStart)
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

            if showVideo or True:
                npframe = np.frombuffer(frame, np.uint8).reshape((height, width, 3)).copy()
            else:
                npframe = np.frombuffer(frame, np.uint8).reshape((height, width, 3))

            # This should get rid of the pvc pipe at the corner taking over the tracking
            npframe[0:150, 0:150, :] = 0
            npframe[0:400, 0:100, :] = 0
            npframe[0:500, 0:50, :] = 0
            npframe[0:25, 0:280, :] = 0
            npframe[:, 1070:, :] = 0
            npframe[0:100, 1010:, :] = 0
            npframe[100:250, 1025:, :] = 0
            npframe[250:450, 1045:, :] = 0
            # There's also some weird red pixels that are just stuck
            npframe[200:220, 130:145, :] = 0
            npframe[780:790, 665:675, :] = 0

            # print("state = {}".format(currentState))

            # =====================
            # Now we're actually looking at a frame and analyzing it!

            if currentState == STATE_START_LIGHTSON:
                # Just waiting for lights off
                brightness = np.sum(npframe)
                if timeStamps[videoFrame] > minLightTimeStamp:
                    if brightness < lightOffThreshold:
                        print("state start lighton --> start lightoff")
                        currentState = STATE_START_LIGHTSOFF
                        x1 = 0
                        x2 = int(width / 2)
                        y1 = 0
                        y2 = int(height / 2)
                        lightsOffFrame = videoFrame

            elif currentState == STATE_START_LIGHTSOFF:
                # Now just waiting for the rat
                # Keep checking upper left part of video for anything above threshold
                x1 = 0
                x2 = int(width / 2)
                y1 = 0
                y2 = int(height / 2)
                rchan = npframe[y1:y2, x1:x2, 2].flatten()
                gchan = npframe[y1:y2, x1:x2, 1].flatten()

                if np.count_nonzero(rchan > threshold) > 0 and np.count_nonzero(gchan > threshold) > 0:
                    # A light!
                    print("state start lightoff --> task")
                    currentState = STATE_TASK
                    hasRed = True
                    hasGreen = True

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
                hasGreen = False
                hasRed = False
                if np.count_nonzero(gchan > threshold) > 0:
                    mgidx = np.argmax(gchan)
                    mgcoords = np.unravel_index(mgidx, (subHeight, subWidth))
                    hasGreen = True

                if np.count_nonzero(rchan > threshold) > 0:
                    mridx = np.argmax(rchan)
                    mrcoords = np.unravel_index(mridx, (subHeight, subWidth))
                    hasRed = True

                if hasGreen and not hasRed:
                    ratXLoc = mgcoords[1] + x1
                    ratYLoc = mgcoords[1] + y1
                elif hasRed and not hasGreen:
                    ratXLoc = mrcoords[1] + x1
                    ratYLoc = mrcoords[1] + y1
                elif hasRed and hasGreen:
                    ratXLoc = int((mrcoords[1] + mgcoords[1]) / 2.0) + x1
                    ratYLoc = int((mrcoords[0] + mgcoords[0]) / 2.0) + y1

                # check for end of task (lights on)
                brightness = np.sum(npframe)
                if brightness > lightOnThreshold:
                    print("state task --> lights on")
                    currentState = STATE_END_LIGHTSON
                    lightsOnFrame = videoFrame

            if showVideo:
                if currentState == STATE_TASK:
                    if hasGreen:
                        for i in range(-3, 3):
                            for j in range(-3, 3):
                                npframe[mgcoords[0] + i + y1, mgcoords[1] + j + x1, 0] = 255
                                npframe[mgcoords[0] + i + y1, mgcoords[1] + j + x1, 1] = 255
                                npframe[mgcoords[0] + i + y1, mgcoords[1] + j + x1, 2] = 0
                    if hasRed:
                        for i in range(-3, 3):
                            for j in range(-3, 3):
                                npframe[mrcoords[0] + i + y1, mrcoords[1] + j + x1, 0] = 255
                                npframe[mrcoords[0] + i + y1, mrcoords[1] + j + x1, 1] = 0
                                npframe[mrcoords[0] + i + y1, mrcoords[1] + j + x1, 2] = 255

                if currentState == STATE_START_LIGHTSOFF or currentState == STATE_TASK:
                    npframe[y1:y2, x1, :] = 255
                    npframe[y1:y2, x2 - 1, :] = 255
                    npframe[y1, x1:x2, :] = 255
                    npframe[y2 - 1, x1:x2, :] = 255

                # frameDur = 1 if currentState == STATE_START_LIGHTSON else 100

                cv2.imshow("frame", npframe)
                if False:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        quitReq = True
                        break
                else:
                    if cv2.waitKey() & 0xFF == ord('q'):
                        quitReq = True
                        break

            outputArray[outputIdx]['timestamp'] = timeStamps[videoFrame]
            outputArray[outputIdx]['x1'] = ratXLoc
            outputArray[outputIdx]['y1'] = ratYLoc
            outputArray[outputIdx]['x2'] = 0
            outputArray[outputIdx]['y2'] = 0
            # print(outputArray[outputIdx])
            outputIdx += 1
            videoFrame += 1

        if quitReq:
            break

    if not quitReq:
        print("saving to file {}...".format(outputFileName))
        outputArray = outputArray[0:outputIdx]
        with open(outputFileName, "wb") as outFile:
            header = "<Start settings>\nsource: TrodesCameraExtractor.py\n" + \
                "lightOnFrame: {}\n".format(lightsOnFrame) + \
                "lightOffFrame: {}\n".format(lightsOffFrame) + \
                "<End settings>\n"
            outFile.write(bytes(header, 'utf-8'))
            outputArray.tofile(outFile)

    # process.wait()
    if showVideo:
        cv2.destroyAllWindows()

    return outputFileName


# def runTest():
#     videoFileName = "/home/wcroughan/Desktop/20211215_140433.1.h264"
#     # videoFileName = "/home/wcroughan/Desktop/20220224_192509.1.h264"

#     # t1 = time.perf_counter()
#     # processRawTrodesVideo(videoFileName, timestampFileName=None,
#     #                       threshold=50, searchDist=150, showVideo=False, frameBatchSize=2000,
#     #                       maxNumBatches=None, outputFileName=None, batchStart=4000, overwriteMode=True)
#     # t2 = time.perf_counter()
#     processRawTrodesVideo(videoFileName, timestampFileName=None,
#                           threshold=50, searchDist=150, showVideo=False, frameBatchSize=1000,
#                           maxNumBatches=None, outputFileName=None, batchStart=4000, overwriteMode=True)
#     # t3 = time.perf_counter()
#     # print("batch size 500: ", t2 - t1)
#     # print("batch size 1000: ", t3 - t2)
#     # batch size 2000 gets killed ... batch size 1000 is twice as fast as 500

#     posFileName = "/home/wcroughan/Desktop/20211215_140433.1.videoPositionTracking"
#     originalPosFileName = "/home/wcroughan/Desktop/originalTracking.videoPositionTracking"

#     settings, position_data = readRawPositionData(posFileName)
#     # print(position_data)
#     print(position_data.shape)
#     xs, ys, ts = processPosData(position_data)
#     # plt.plot(xs, ys)
#     plt.plot(ts, xs)
#     plt.xlim(ts[0], ts[-1])

#     settings, position_data = readRawPositionData(originalPosFileName)
#     # print(position_data)
#     print(position_data.shape)
#     xs, ys, ts = processPosData(position_data)
#     # plt.plot(xs, ys)
#     plt.plot(ts, xs)
#     plt.legend(["py", "trodes"])
#     plt.show()


def runAllBradTask():
    # dataDir = "/media/WDC6/"
    # animalNames = ["B13", "B14"]
    dataDir = "/media/WDC8/"
    animalNames = ["B18"]
    allTrodesVids = []
    for animalName in animalNames:
        gl = dataDir + "/" + animalName + "/bradtasksessions/*/*.h264"
        videoNameList = glob.glob(gl)
        allTrodesVids += videoNameList
    print("\n".join(allTrodesVids))

    for videoName in allTrodesVids:
        pass
        # print(f"========================================\n\nRunning {videoName}"
        # "\n\n==========================================")
        # processRawTrodesVideo(videoName, overwriteMode="never")
        # getTrodesLightTimes(videoFileName=videoName, showVideo=False, initialSkipAmt=5)

    usbDataDir = "/media/WDC7/videos/B16-20/trimmed/"
    allUSBVids = []
    for animalName in animalNames:
        gl = usbDataDir + "/" + animalName + "/*.mkv"
        videoNameList = glob.glob(gl)
        allUSBVids += videoNameList
    print("\n".join(allUSBVids))
    for videoName in allUSBVids:
        processUSBVideoData(videoName, showVideo=True, overwriteMode="always")


def rerunTrodesVideos(showVideo=False):
    animalNames = ["B13", "B14"]
    # animalNames = ["B13"]
    allVids = []
    for animalName in animalNames:
        info = getInfoForAnimal(animalName)
        for s in info.rerun_trodes_videos:
            vidFile = getTrodesVideoFile(s, info.data_dir)
            allVids.append(vidFile)

    for videoName in allVids:
        print(
            f"========================================\n\nRunning {videoName}"
            "\n\n==========================================")
        if showVideo:
            processRawTrodesVideo(videoName, overwriteMode="ask", showVideo=True)
        else:
            processRawTrodesVideo(videoName, overwriteMode="always")


def rerunUSBVideos(showVideo=False):
    animalNames = ["B13", "B14"]
    # animalNames = ["B13"]
    allVids = []
    for animalName in animalNames:
        info = getInfoForAnimal(animalName)
        possibleDirectories = [
            "/media/WDC6/{}/".format(animalName), "/media/WDC7/{}/".format(animalName)]
        for s in info.rerun_usb_videos:
            vidFile = getUSBVideoFile(s, possibleDirectories)
            allVids.append(vidFile)

    for videoName in allVids:
        print(
            f"========================================\n\nRunning {videoName}"
            "\n\n==========================================")
        if showVideo:
            processUSBVideoData(videoName, overwriteMode="ask", showVideo=True)
        else:
            res = processUSBVideoData(videoName, overwriteMode="always")
            if res[0] is None or res[1] is None:
                print(f"================================================================================"
                      f"{videoName} should be excluded"
                      f"==========================================")


def getFrameColorHeuristic(frame, greyDiffThreshLow=25, greyDiffThreshHigh=200):
    i21, i22 = frame.shape[0] // 2, frame.shape[1] // 2
    diff1 = frame[:i21, :i22, 0] - frame[:i21, :i22, 1]
    diff2 = frame[:i21, :i22, 0] - frame[:i21, :i22, 2]
    diff3 = frame[:i21, :i22, 1] - frame[:i21, :i22, 2]
    h1 = np.histogram(diff1, bins=[0, greyDiffThreshLow, greyDiffThreshHigh, 1000])
    h2 = np.histogram(diff2, bins=[0, greyDiffThreshLow, greyDiffThreshHigh, 1000])
    h3 = np.histogram(diff3, bins=[0, greyDiffThreshLow, greyDiffThreshHigh, 1000])
    # return np.count_nonzero(np.logical_and(
    # gdiff > greyDiffThreshLow, gdiff < greyDiffThreshHigh))
    return h1[0], h2[0], h3[0]


def findLightTimesLinear(videoFileName, frameStart, frameEnd, frameStride, heuristicFunction, findLightOff=True,
                         findLightOn=True, showVideo=False, lightOnThreshold=None, lightOffThreshold=None,
                         ignoreFirstFrames=None):
    """
    making a separate function here b/c hopefully that means freed memory?
    """
    probe = ffmpeg.probe(videoFileName)
    video_stream = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    if frameEnd is None:
        batch = getFrameBatch(videoFileName, frameStart, frameStride=frameStride)
    else:
        nFrames = (frameEnd - frameStart) // frameStride
        batch = getFrameBatch(videoFileName, frameStart, numFrames=nFrames, frameStride=frameStride)
    frames = np.frombuffer(batch, np.uint8).reshape((-1, height, width, 3))
    nFrames = frames.shape[0]

    allFrameH = np.array(
        [heuristicFunction(frames[i, :, :, :].reshape(height, width, 3)) for i in range(nFrames)])
    # allFrameH = np.array([getFrameColorHeuristic(
    # frames[i, :, :, :].reshape(height, width, 3))[1][1] for i in range(nFrames)])
    if findLightOff and findLightOn:
        nhalf = len(allFrameH) // 2
        maxH1 = np.max(allFrameH[0:nhalf])
        minH1 = np.min(allFrameH[0:nhalf])
        maxH2 = np.max(allFrameH[nhalf:])
        minH2 = np.min(allFrameH[nhalf:])
        maxH = min(maxH1, maxH2)
        minH = max(minH1, minH2)
        thresh = (maxH + minH) / 2.0

        # Check that there's frames with both light on and off at start and finish
        d1 = maxH1 - minH1
        d2 = maxH2 - minH2
        dd = d1 / d2 + d2 / d1
        print("light stats:")
        print("\t1st half:\n\t\t{}\n\t\t{}\n\t\t({})".format(minH1, maxH1, d1))
        print("\t2nd half:\n\t\t{}\n\t\t{}\n\t\t({})".format(minH2, maxH2, d2))
        print("\tdd = " + str(dd))
        if dd > 10:
            # print("WARNING: might not have light on or off times, autoEnabling video display")
            # showVideo = True
            print("WARNING: might not have light on or off times, using just the overall light mins and maxes")
            maxH = max(maxH1, maxH2)
            minH = min(minH1, minH2)
            thresh = (maxH + minH) / 2.0

    else:
        maxH = np.max(allFrameH)
        minH = np.min(allFrameH)
        thresh = (maxH + minH) / 2.0

    if lightOffThreshold is not None:
        lightOffThreshold *= 255 * width * height
    else:
        lightOffThreshold = thresh
    if lightOnThreshold is not None:
        lightOnThreshold *= 255 * width * height
    else:
        lightOnThreshold = thresh

    if showVideo and False:
        # for frameI in range(nFrames):
        #     if frameI < 15 or nFrames - frameI < 15:
        #         frame = frames[frameI, :, :, :].reshape((height, width, 3))
        #         heur = getFrameColorHeuristic(frame)
        #         print(heur)

        for frameI in range(nFrames):
            if frameI < 15 or nFrames - frameI < 15:
                frame = frames[frameI, :, :, :].reshape((height, width, 3))
                heur = getFrameColorHeuristic(frame)
                frame = cv2.putText(frame, "{}: h={}".format(frameI, heur), (100, 100),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
                # frame = cv2.putText(frame, "thresh={}, loffF2={}, lonF2={}".format(thresh, loffF2, lonF2), (100, 200),
                # cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
                cv2.imshow("frame", frame)
                if cv2.waitKey() & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()

    if showVideo:
        print(allFrameH)
        if findLightOff:
            print("loff thresh", lightOffThreshold)
        if findLightOn:
            print("lon thresh", lightOnThreshold)

    loffSet = False
    lonSet = False
    loffF1 = None
    loffF2 = None
    lonF1 = None
    lonF2 = None

    for i in range(1, nFrames):
        if ignoreFirstFrames is not None and i * frameStride < ignoreFirstFrames:
            # print(f"ignoring frame {i} (stride={frameStride}, product={i*frameStride}), "
            #       f"skpping first {ignoreFirstFrames}")
            continue

        if allFrameH[i] < lightOffThreshold and allFrameH[i - 1] > lightOffThreshold and not loffSet and findLightOff:
            loffF1 = i - 1
            loffF2 = i
            loffSet = True
            if not findLightOn:
                break
        if allFrameH[i] > lightOnThreshold and allFrameH[i - 1] < lightOnThreshold and findLightOn:
            lonF1 = i - 1
            lonF2 = i
            lonSet = True
            if not findLightOff:
                break

    if not lonSet and findLightOn:
        lonF1 = nFrames - 1
        lonF2 = nFrames

    if showVideo and True:
        # for frameI in range(nFrames):
        #     if frameI < 15 or nFrames - frameI < 15:
        #         frame = frames[frameI, :, :, :].reshape((height, width, 3))
        #         heur = getFrameColorHeuristic(frame)
        #         print(heur)
        print("thresh={}, loffF2={}, lonF2={}".format(thresh, loffF2, lonF2))

        for frameI in range(nFrames):
            frame = frames[frameI, :, :, :].reshape((height, width, 3))
            # heur = getFrameColorHeuristic(frame)
            heur = allFrameH[frameI]
            frame = cv2.putText(frame, "{}: h={}".format(frameI, heur), (100, 100),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0))
            frame = cv2.putText(frame, "thresh={}, loffF2={}, lonF2={}".format(thresh, loffF2, lonF2), (100, 200),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0))
            cv2.imshow("frame", frame)
            if cv2.waitKey() & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    return loffF1, loffF2, lonF1, lonF2


def findLightTimesLinearUSB(videoFileName, frameStart, frameEnd, frameStride, findLightOff=True, findLightOn=True,
                            showVideo=False, lightOnThreshold=None, lightOffThreshold=None):
    def hfunc(frame):
        return getFrameColorHeuristic(frame)[1][1]
    return findLightTimesLinear(videoFileName, frameStart, frameEnd, frameStride, hfunc, findLightOff=findLightOff,
                                findLightOn=findLightOn, showVideo=showVideo, lightOnThreshold=lightOnThreshold,
                                lightOffThreshold=lightOffThreshold)


def findLightTimesLinearTrodes(videoFileName, frameStart, frameEnd, frameStride, findLightOff=True, findLightOn=True,
                               showVideo=False, lightOnThreshold=None, lightOffThreshold=None, ignoreFirstFrames=None):
    def hfunc(frame):
        return np.sum(frame)
    return findLightTimesLinear(videoFileName, frameStart, frameEnd, frameStride, hfunc, findLightOff=findLightOff,
                                findLightOn=findLightOn, showVideo=showVideo, lightOnThreshold=lightOnThreshold,
                                lightOffThreshold=lightOffThreshold, ignoreFirstFrames=ignoreFirstFrames)


def processUSBVideoData(videoFileName, batchStart=0, frameBatchSize=1000, greyDiffThreshHigh=100, greyDiffThreshLow=25,
                        lightOnThreshold=0.1, lightOffThreshold=0.04, maxNumBatches=None, showVideo=False,
                        outputFileName=None, overwriteMode="ask", initialSkipAmt=8):

    if outputFileName is None:
        outputFileName = videoFileName + ".usbLightTimes"

    if os.path.exists(outputFileName):
        if overwriteMode == "never":
            return

        elif overwriteMode == "rename":
            outputFileName += ".again"

        elif overwriteMode == "ask":
            c = input("Output file {} exists, overwrite? (y/N/l):".format(outputFileName))
            if c == "l":
                print(outputFileName)
                retArr = np.fromfile(outputFileName, dtype=int, sep=",")
                print(retArr)
                return retArr[0], retArr[1]
            elif c != "y":
                return

        elif overwriteMode == "loadOld":
            retArr = np.fromfile(outputFileName, dtype=int, sep=",")
            return retArr[0], retArr[1]

        elif overwriteMode != "always":
            print("Unknown overwrite mode {}".format(overwriteMode))
            return

    # probe = ffmpeg.probe(videoFileName)
    # video_stream = next(
    #     (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    # width = int(video_stream['width'])
    # height = int(video_stream['height'])

    lightsOffFrame = None
    lightsOnFrame = None

    initialFrameJump = np.power(2, initialSkipAmt)

    # Step one: find left and right bound on light off and on time
    loffF1, loffF2, lonF1, lonF2 = findLightTimesLinearUSB(
        videoFileName, batchStart, None, initialFrameJump, showVideo=showVideo)
    if loffF1 is None or loffF2 is None or lonF1 is None or lonF2 is None:
        return None, None
    loffF1 *= initialFrameJump
    loffF2 *= initialFrameJump
    lonF1 *= initialFrameJump
    lonF2 *= initialFrameJump
    loffF1 += batchStart
    loffF2 += batchStart
    lonF1 += batchStart
    lonF2 += batchStart

    print(loffF1, loffF2, lonF1, lonF2)

    # Step two: for each of those intervals, get the video in that range and binary search that ish
    _, lightsOffFrame, _, _ = findLightTimesLinearUSB(
        videoFileName, loffF1, loffF2, 1, showVideo=showVideo, findLightOn=False)
    lightsOffFrame += loffF1
    _, _, _, lightsOnFrame = findLightTimesLinearUSB(
        videoFileName, lonF1, lonF2, 1, showVideo=showVideo, findLightOff=False)
    lightsOnFrame += lonF1

    print(lightsOffFrame, lightsOnFrame)
    if lightsOffFrame is None or lightsOnFrame is None:
        return None, None

    outArr = np.array([lightsOffFrame, lightsOnFrame])
    with open(outputFileName, "w") as outFile:
        outArr.tofile(outFile, sep=",")

    return lightsOffFrame, lightsOnFrame


def playFrames(videoFileName, frameStart, frameEnd):
    probe = ffmpeg.probe(videoFileName)
    video_stream = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    batch = getFrameBatch(videoFileName, frameStart, numFrames=frameEnd - frameStart)
    batchBufferIdx = 0
    batchBufferIdxJump = width * height * 3
    quitReq = False
    while True:
        frame = batch[batchBufferIdx:batchBufferIdx + batchBufferIdxJump]
        batchBufferIdx += batchBufferIdxJump
        if not frame:
            break

        npframe = np.frombuffer(frame, np.uint8).reshape((height, width, 3))
        cv2.imshow("frame", npframe)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            quitReq = True
            break

    if not quitReq:
        cv2.waitKey()

    cv2.destroyAllWindows()


# def testUSBAlignment():
#     usbVideoFile = "/home/wcroughan/Desktop/2021-12-09_18-04-47.mkv"
#     usbLightOffFrame, usbLightOnFrame = processUSBVideoData(
#         usbVideoFile, overwriteMode="loadOld", showVideo=False)
#     print(usbLightOffFrame, usbLightOnFrame)
#     # playFrames(usbVideoFile, usbLightOffFrame - 20, usbLightOffFrame + 20)
#     # playFrames(usbVideoFile, usbLightOnFrame - 20, usbLightOnFrame + 20)
#     sesh_name = "20211209_180352"
#     dataFileName = "/media/WDC7/B13/processed_data/B13_bradtask.dat"
#     ratData = BTData()
#     ratData.loadFromFile(dataFileName)
#     sesh = ratData.getSessions(lambda s: s.name == sesh_name)[0]
#     trodesVideoFile = "/media/WDC6/B13/bradtasksessions/20211209_180352/20211209_180352.1.h264"
#     trodesOutputFileName = '.'.join(trodesVideoFile.split('.')[0:-1]) + '.videoPositionTracking.tce'
#     processedTrodesVideoFileName = processRawTrodesVideo(
#         trodesVideoFile, overwriteMode="never", outputFileName=trodesOutputFileName)
#     print("trodesVideoOutput in {}".format(processedTrodesVideoFileName))
#     settings, position_data = readRawPositionData(processedTrodesVideoFileName)

#     trodesLightOnFrame = None
#     trodesLightOffFrame = None
#     for s in settings:
#         s = str(s).lower()
#         print(s)
#         if "lightonframe" in s:
#             ss = s.split(":")[1].strip()
#             if ss[-3:] == "\\n'":
#                 ss = ss[0:-3]
#             print(ss)
#             trodesLightOnFrame = int(ss)
#         if "lightoffframe" in s:
#             ss = s.split(":")[1].strip()
#             if ss[-3:] == "\\n'":
#                 ss = ss[0:-3]
#             print(ss)
#             trodesLightOffFrame = int(ss)
#     print(trodesLightOnFrame, trodesLightOffFrame)

#     trodesLightOnTime = position_data[trodesLightOnFrame]['timestamp']
#     trodesLightOffTime = position_data[trodesLightOffFrame]['timestamp']
#     print(trodesLightOnTime, trodesLightOffTime)

#     bufferTime = 1

#     for wft, wlt in reversed(list(zip(sesh.home_well_find_times, sesh.home_well_leave_times))):
#         t1 = wft - bufferTime * TRODES_SAMPLING_RATE
#         t2 = wlt + bufferTime * TRODES_SAMPLING_RATE
#         print(t1, t2)

#         wellFrames = np.floor(np.interp([t1, t2], [trodesLightOffTime, trodesLightOnTime],
#                                         [usbLightOffFrame, usbLightOnFrame])).astype(int)

#         playFrames(usbVideoFile, wellFrames[0], wellFrames[1])


def runBatchTest():
    videoFileName = "/home/wcroughan/Desktop/20220224_192509.1.h264"
    b = getFrameBatch(videoFileName, 0, numFrames=100, frameStride=1)
    print(b)


if __name__ == "__main__":
    # rerunUSBVideos(False)
    # rerunTrodesVideos(False)
    # videoName = "/media/WDC6/B13/bradtasksessions/20220308_140707/20220308_140707.1.h264"
    # processRawTrodesVideo(videoName, overwriteMode="always", showVideo=True, batchStart=15*12*60)

    # runAllBradTask()

    toff, ton = getTrodesLightTimes(
        "/media/WDC8/B16/bradtasksessions/20221113_105126/20221113_105126.1.h264", showVideo=True)

    uoff, uon = processUSBVideoData(
        "/media/WDC8/videos/labvideos/trimmed/B16/20221113_1.mkv", overwriteMode="rename", showVideo=True)
