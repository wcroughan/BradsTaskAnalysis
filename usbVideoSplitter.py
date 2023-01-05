import os


def splitVideo(inFile, startTimeCode, endTimeCode, outFile):
    if startTimeCode is None:
        assert endTimeCode is None
        cmd = f"ffmpeg -ss {startTimeCode} -to {endTimeCode} -i {inFile} -an -muxdelay 0 -muxpreload 0 {outFile}"
    else:
        assert endTimeCode is not None
        cmd = f"ffmpeg -i {inFile} -an -muxdelay 0 -muxpreload 0 {outFile}"

    print(cmd)
    # os.system(cmd)


def splitAllVideos(cutTimesFile, videoDirectory):
    with open(cutTimesFile) as f:
        while True:
            sessionName = ""
            inFileName = ""
            l = ""
            eof = False
            while l == "":
                l = f.readline()
                if l == "":
                    # end of file
                    eof = True
                    break
                l = l.strip()

            if eof:
                break

            assert l[0:6] == "202211"
            sessionName = l
            l = f.readline().strip()
            ratOrder = l.split(", ")
            ratI = 0
            while ratI < len(ratOrder):
                timeCodes = None
                l = f.readline().strip()
                if l.count("-") == 4:
                    inFileName = os.path.join(videoDirectory, l + ".mkv")
                elif l.count("-") == 1:
                    timeCodes = l.split("-")
                    if timeCodes[0].count(":") == 1:
                        timeCodes[0] = "0:" + timeCodes[0]
                    if timeCodes[1].count(":") == 1:
                        timeCodes[1] = "0:" + timeCodes[1]
                elif l == "wholefile":
                    timeCodes = [None, None]
                else:
                    assert False

                if timeCodes is not None:
                    ratName = "B" + ratOrder[ratI]
                    saveLocation = os.path.join(os.path.dirname(inFileName), "trimmed", ratName)
                    if not os.path.exists(saveLocation):
                        os.makedirs(saveLocation)
                    outFileName = os.path.join(saveLocation, sessionName + ".mkv")
                    splitVideo(inFileName, timeCodes[0], timeCodes[1], outFileName)
                    ratI += 1


if __name__ == "__main__":
    cutTimesFile = "../labnotes/videocuttimes.txt"
    videoDirectory = "/media/WDC8/videos/labvideos"
    splitAllVideos(cutTimesFile, videoDirectory)
