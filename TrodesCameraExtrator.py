# Trodes pausing when it loses LEDs is annoying so I'm just writing my own version

import io
import ffmpeg
import numpy as np
import cv2


# inputFileName = "/home/wcroughan/Desktop/20220224_192509.1.h264"
inputFileName = "/home/wcroughan/Desktop/20220309_173701.1.h264"
outputFileName = "/home/wcroughan/Desktop/output.txt"

probe = ffmpeg.probe(inputFileName)
video_stream = next(
    (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
width = int(video_stream['width'])
height = int(video_stream['height'])


def getFrameBatch(startFrame, numFrames=100):
    process = (
        ffmpeg
        .input(inputFileName)
        .video
        .filter('select', 'gte(n,{})'.format(startFrame))
        .output('pipe:', format='rawvideo', vframes=numFrames, pix_fmt='rgb24')
        .run_async(pipe_stdin=True, pipe_stdout=True)
    )

    stream = io.BytesIO()
    stream.seek(0)

    outs, errs = process.communicate(input=stream.getvalue())
    return outs


quitReq = False
batchStart = 4000
batchSize = 100
maxNumBatches = 3
batchNum = -1
while True:
    batchNum += 1
    if batchNum > maxNumBatches:
        break

    batch = getFrameBatch(batchStart, numFrames=batchSize)
    batchStart += batchSize

    print("hello!!!!")
    print("hello!!!!")
    # print(batch)

    if not batch:
        break

    batchBufferIdx = 0
    batchBufferIdxJump = width * height * 3
    while True:
        frame = batch[batchBufferIdx:batchBufferIdx + batchBufferIdxJump]
        batchBufferIdx += batchBufferIdxJump

        print(batchBufferIdx)

        if not frame:
            break

        npframe = np.frombuffer(frame, np.uint8).reshape((height, width, 3)).copy()
        # npframe[:, :, 0] = 0
        # npframe[:, :, 2] = 0
        rchan = npframe[:, :, 2].flatten()
        gchan = npframe[:, :, 1].flatten()

        mgidx = np.argmax(gchan)
        mgcoords = np.unravel_index(mgidx, (height, width))
        for i in range(-3, 3):
            for j in range(-3, 3):
                npframe[mgcoords[0] + i, mgcoords[1] + j, :] = [1, 0, 0]

        mridx = np.argmax(rchan)
        mrcoords = np.unravel_index(mridx, (height, width))
        for i in range(-3, 3):
            for j in range(-3, 3):
                npframe[mrcoords[0] + i, mrcoords[1] + j, :] = [1, 0, 0]

        cv2.imshow("frame", npframe)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            quitReq = True
            break
    if quitReq:
        break

# process.wait()
cv2.destroyAllWindows()
