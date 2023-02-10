import numpy as np
import os
from BTSession import BTSession
from BTSession import BehaviorPeriod as BP
from BTData import BTData
from PlotUtil import setupBehaviorTracePlot
from UtilFunctions import getLoadInfo
from consts import TRODES_SAMPLING_RATE
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
import matplotlib as mpl
from multiprocessing import Pool


def getFrame(sesh: BTSession, DIST_FACTOR, VEL_FACTOR, resolution, frame):
    frameStartTime, frameEndTime = frame
    # pre-evaluate bp for efficiency
    keepFlags1 = sesh.evalBehaviorPeriod(
        BP(True, timeInterval=[frameStartTime, frameEndTime, "posIdx"], inclusionFlags="moving"))
    bp1 = BP(True, inclusionArray=keepFlags1)
    return sesh.getValueMap(lambda pos: sesh.getDotProductScore(
        bp1, pos, distanceWeight=DIST_FACTOR, velocityWeight=VEL_FACTOR), resolution=resolution).T


def makeAnimation(sesh: BTSession, saveFile=None):
    FRAME_RATE = 10
    VIDEO_SPEED = 2
    PLOT_LEN = 0.5
    TSTART = 25
    TEND = 45
    x = sesh.probePosXs
    y = sesh.probePosYs
    mv = sesh.probeIsMv
    bout = sesh.probeBoutCategory
    vel = sesh.probeVelCmPerS
    smvel = sesh.probeSmoothVel
    t = sesh.probePos_ts / TRODES_SAMPLING_RATE
    t = t - t[0]

    resolution = 21
    DIST_FACTOR = 0.0
    VEL_FACTOR = 0.0

    if TEND is None:
        TEND = t[-1]

    frameStartTimes = np.arange(TSTART, TEND - PLOT_LEN, VIDEO_SPEED / FRAME_RATE)
    frameEndTimes = frameStartTimes + PLOT_LEN
    frameStarts_posIdx = np.searchsorted(t, frameStartTimes)
    frameEnds_posIdx = np.searchsorted(t, frameEndTimes)
    frames = list(zip(frameStarts_posIdx, frameEnds_posIdx, range(len(frameEnds_posIdx))))
    cumFrames = list(zip([0] * len(frameEnds_posIdx), frameEnds_posIdx))
    curFrames = list(zip(frameStarts_posIdx, frameEnds_posIdx))

    # cumulativeValImgs = np.empty((len(frames), resolution, resolution))
    # currentValImgs = np.empty((len(frames), resolution, resolution))
    cumulativeValImgs = [np.empty((resolution, resolution)) for _ in range(len(frames))]
    currentValImgs = [np.empty((resolution, resolution)) for _ in range(len(frames))]
    with Pool() as p:
        currentValImgs = p.map(partial(getFrame, sesh, DIST_FACTOR,
                               VEL_FACTOR, resolution), curFrames)
        cumulativeValImgs = p.map(partial(getFrame, sesh, DIST_FACTOR,
                                  VEL_FACTOR, resolution), cumFrames)

    # for fi in range(len(frames)):
    #     print("frame %d of %d" % (fi, len(frames)) + " " * 20, end="\r")
    #     cumulativeValImgs[fi] = getFrame(sesh, DIST_FACTOR, VEL_FACTOR, resolution, cumFrames[fi]).T
    #     currentValImgs[fi] = getFrame(sesh, DIST_FACTOR, VEL_FACTOR,  resolution, curFrames[fi]).T
    # print(" " * 100, end="\r")
    input("press enter to continue")

    cumulativeValImgs = np.array(cumulativeValImgs)
    currentValImgs = np.array(currentValImgs)
    cumulativeValMin = np.nanmin(cumulativeValImgs)
    cumulativeValMax = np.nanmax(cumulativeValImgs)
    currentValMin = np.nanmin(currentValImgs)
    currentValMax = np.nanmax(currentValImgs)

    xmv1 = x.copy()
    xmv1[~mv] = np.nan
    xmv2 = x.copy()
    xmv2[mv] = np.nan
    ymv1 = y.copy()
    ymv1[~mv] = np.nan
    ymv2 = y.copy()
    ymv2[mv] = np.nan

    xbo1 = x.copy()
    xbo1[bout != BTSession.BOUT_STATE_EXPLORE] = np.nan
    xbo2 = x.copy()
    xbo2[bout != BTSession.BOUT_STATE_REST] = np.nan
    xbo3 = x.copy()
    xbo3[bout != BTSession.BOUT_STATE_REWARD] = np.nan
    ybo1 = y.copy()
    ybo1[bout != BTSession.BOUT_STATE_EXPLORE] = np.nan
    ybo2 = y.copy()
    ybo2[bout != BTSession.BOUT_STATE_REST] = np.nan
    ybo3 = y.copy()
    ybo3[bout != BTSession.BOUT_STATE_REWARD] = np.nan

    fig, axs = plt.subplots(2, 2)
    p11, = axs[0, 0].plot([])
    p12, = axs[0, 0].plot([])
    p21, = axs[0, 1].plot([])
    p22, = axs[0, 1].plot([])
    p31, = axs[1, 0].plot([])
    p32, = axs[1, 0].plot([])
    im1 = axs[0, 0].imshow(cumulativeValImgs[0, :, :], cmap=mpl.colormaps["coolwarm"],
                           vmin=cumulativeValMin, vmax=cumulativeValMax,
                           interpolation="nearest", extent=(0, 6, 0, 6),
                           origin="lower")
    im2 = axs[0, 1].imshow(currentValImgs[0, :, :], cmap=mpl.colormaps["coolwarm"],
                           vmin=currentValMin, vmax=currentValMax,
                           interpolation="nearest", extent=(0, 6, 0, 6),
                           origin="lower")

    axs[0, 0].set_title("cumulative")
    axs[0, 1].set_title("current")

    # c1, = axs[1, 0].plot(
    #     t[frames[0][0]:frames[0][1]], mv[frames[0][0]:frames[0][1]])
    # c2, = axs[1, 0].plot(
    #     t[frames[0][0]:frames[0][1]], bout[frames[0][0]:frames[0][1]])
    v1, = axs[1, 1].plot(
        t[frames[0][0]:frames[0][1]], vel[frames[0][0]:frames[0][1]])
    v2, = axs[1, 1].plot(
        t[frames[0][0]:frames[0][1]], smvel[frames[0][0]:frames[0][1]])
    axs[1, 1].set_ylim(0, np.max(vel))

    p11.set_animated(True)
    p12.set_animated(True)
    p21.set_animated(True)
    p22.set_animated(True)
    p31.set_animated(True)
    p32.set_animated(True)
    # c1.set_animated(True)
    # c2.set_animated(True)
    v1.set_animated(True)
    v2.set_animated(True)
    im1.set_animated(True)
    im2.set_animated(True)

    setupBehaviorTracePlot(axs[0, 0], sesh)
    setupBehaviorTracePlot(axs[0, 1], sesh)
    setupBehaviorTracePlot(axs[1, 0], sesh)

    def animFunc(fs):
        p11.set_data(xmv1[fs[0]:fs[1]], ymv1[fs[0]:fs[1]])
        p12.set_data(xmv2[fs[0]:fs[1]], ymv2[fs[0]:fs[1]])
        p21.set_data(xmv1[fs[0]:fs[1]], ymv1[fs[0]:fs[1]])
        p22.set_data(xmv2[fs[0]:fs[1]], ymv2[fs[0]:fs[1]])
        p31.set_data(xmv1[fs[0]:fs[1]], ymv1[fs[0]:fs[1]])
        p32.set_data(xmv2[fs[0]:fs[1]], ymv2[fs[0]:fs[1]])
        # c1.set_data(t[fs[0]:fs[1]], mv[fs[0]:fs[1]])
        # c2.set_data(t[fs[0]:fs[1]], bout[fs[0]:fs[1]])
        im1.set_data(cumulativeValImgs[fs[2], :, :])
        im2.set_data(currentValImgs[fs[2], :, :])

        # axs[1, 0].set_xlim(t[fs[0]], t[fs[1]])
        v1.set_data(t[fs[0]:fs[1]], vel[fs[0]:fs[1]])
        v2.set_data(t[fs[0]:fs[1]], smvel[fs[0]:fs[1]])
        axs[1, 1].set_xlim(t[fs[0]], t[fs[1]])
        return p11, p12, p21, p22, v1, v2, im1, p31, p32, im2

    ani = FuncAnimation(fig, animFunc, frames, repeat=False,
                        interval=1000/FRAME_RATE, blit=True,
                        init_func=partial(animFunc, frames[0]))

    # start_time = time.perf_counter()
    if saveFile is not None:
        ani.save(saveFile)
    else:
        plt.show()
    # end_time = time.perf_counter()
    # runTime = end_time - start_time
    # print(f"{runTime = }")
    # animFuncRunTime = animFunc.totalTime
    # print(f"{animFuncRunTime = }")


def main():
    ratName = "B17"

    animalInfo = getLoadInfo(ratName)
    dataFilename = os.path.join(animalInfo.output_dir, animalInfo.out_filename)
    print("loading from " + dataFilename)
    ratData = BTData()
    ratData.loadFromFile(dataFilename)
    makeAnimation(ratData.getSessions()[10])


if __name__ == "__main__":
    main()
