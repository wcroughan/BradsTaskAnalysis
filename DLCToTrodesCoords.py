from glob import glob
from scipy.ndimage import gaussian_filter1d, binary_dilation
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib
import os
matplotlib.use('tkagg')


def getDVecsFromWellCoords(wellLocations):
    camVecs = np.zeros((3, 3))
    camVecs[0, 0:2] = wellLocations.loc[(35,), ("x", "y")].to_numpy()[0] - \
        wellLocations.loc[(11,), ("x", "y")].to_numpy()[0]
    camVecs[1, 0:2] = wellLocations.loc[(38,), ("x", "y")].to_numpy()[0] - \
        wellLocations.loc[(11,), ("x", "y")].to_numpy()[0]
    camVecs[2, 0:2] = wellLocations.loc[(14,), ("x", "y")].to_numpy()[0] - \
        wellLocations.loc[(11,), ("x", "y")].to_numpy()[0]
    # camVecs[0,:] = vector from well 11 to well 35
    # camVecs[1,:] = vector from well 11 to well 38
    # camVecs[2,:] = vector from well 11 to well 14
    return camVecs


def projectCamCoords(dat, camNum, sessionName):
    wellLocationFile = "/media/WDC6/DLC/well_locations.csv"
    wellLocations = pd.read_csv(wellLocationFile, header=None,
                                names=["session", "cam", "well", "x", "y"], index_col=["session", "cam", "well"])
    try:
        wellLocations = wellLocations.loc[sessionName]
    except KeyError:
        wellLocations = wellLocations.loc["20210903_180032"]
    # print(wellLocations)

    cam3Locs = wellLocations.loc[3].to_numpy().astype(np.float32)
    camLocs = wellLocations.loc[camNum].to_numpy().astype(np.float32)
    # Need to compensate for the videos being side by side here but not in DLC coord output
    if camNum == 2:
        camLocs[:, 0] -= 376
    elif camNum == 3:
        camLocs[:, 0] -= 708
    cam3Locs[:, 0] -= 708

    # print("camlocs", camLocs)
    transform = cv2.getPerspectiveTransform(camLocs, cam3Locs)
    if False:
        print("===============================")
        print(camLocs)
        withOnes = np.hstack((camLocs, np.ones((4, 1))))
        print(withOnes)
        newLocs = np.matmul(transform, withOnes.T)
        print(newLocs)
        newLocs = newLocs / newLocs[-1, :]
        print(newLocs.T)
        print(cam3Locs)
        print("===============================")

    coords = dat.to_numpy(copy=True)
    coords[:, 2] = 1
    newCoords = np.matmul(transform, coords.T)
    # print("non normal", newCoords.T)
    newCoords = newCoords / newCoords[-1, :]
    # print("normal", newCoords.T)
    newCoords[-1, :] = dat.to_numpy()[:, -1]
    # print("with liklihood", newCoords.T)
    return newCoords.T


def convertDLCToTrodesCoords(cam1Coords, cam2Coords, cam3Coords, sessionName,
                             transform1=None, transform2=None, showPlots=True):
    cam1TrodesCoords = projectCamCoords(cam1Coords, 1, sessionName)
    cam2TrodesCoords = projectCamCoords(cam2Coords, 2, sessionName)
    cam3TrodesCoords = projectCamCoords(cam3Coords, 3, sessionName)

    x1 = cam1TrodesCoords[:, 0]
    x2 = cam2TrodesCoords[:, 0]
    x3 = cam3TrodesCoords[:, 0]
    y1 = cam1TrodesCoords[:, 1]
    y2 = cam2TrodesCoords[:, 1]
    y3 = cam3TrodesCoords[:, 1]
    l1 = cam1TrodesCoords[:, 2]
    l2 = cam2TrodesCoords[:, 2]
    l3 = cam3TrodesCoords[:, 2]

    if showPlots:
        plt.subplot(221)
        plt.plot(x1, y1)
        plt.plot(x2, y2)
        plt.plot(x3, y3)
        plt.title("Perspective shifted")

    # # Just some eyeball compensation based on ouptut
    # x2 -= 25
    # x3 -= 15

    # lthresh = 0.9999
    # x1[l1 < lthresh] = np.nan
    # x2[l2 < lthresh] = np.nan
    # x3[l3 < lthresh] = np.nan
    # y1[l1 < lthresh] = np.nan
    # y2[l2 < lthresh] = np.nan
    # y3[l3 < lthresh] = np.nan

    if transform1 is None:
        assert transform2 is None
        mx1 = np.nanpercentile(x3, 40)
        mx2 = np.nanpercentile(x3, 60)
        my1 = np.nanpercentile(y3, 40)
        my2 = np.nanpercentile(y3, 60)
        # print(f"mx1={mx1}, mx2={mx2}, my1={my1}, my2={my2}, ")
        # print(np.nanpercentile(x1, np.linspace(0, 100, 11)))
        # print(np.nanpercentile(x2, np.linspace(0, 100, 11)))
        # print(np.nanpercentile(x3, np.linspace(0, 100, 11)))

        mx0 = np.nanpercentile(x3, 0)
        mx3 = np.nanpercentile(x3, 100)
        my0 = np.nanpercentile(y3, 0)
        my3 = np.nanpercentile(y3, 100)

        dThresh = 50

        jumpi1 = np.diff(np.vstack((x1, y1)).T, axis=0, prepend=np.zeros((1, 2))) > dThresh
        jumpi1 = np.any(jumpi1, axis=1)
        jumpi1 = binary_dilation(jumpi1, iterations=10)

        jumpi2 = np.diff(np.vstack((x2, y2)).T, axis=0, prepend=np.zeros((1, 2))) > dThresh
        jumpi2 = np.any(jumpi2, axis=1)
        jumpi2 = binary_dilation(jumpi2, iterations=10)

        jumpi3 = np.diff(np.vstack((x3, y3)).T, axis=0, prepend=np.zeros((1, 2))) > dThresh
        jumpi3 = np.any(jumpi3, axis=1)
        jumpi3 = binary_dilation(jumpi3, iterations=10)

        jumpAny = jumpi1 | jumpi2 | jumpi3

        q11i = (x3 < mx1) & (y3 < my1) & ~ jumpAny
        q12i = (x3 < mx1) & (y3 > my2) & ~ jumpAny
        q21i = (x3 > mx2) & (y3 < my1) & ~ jumpAny
        q22i = (x3 > mx2) & (y3 > my2) & ~ jumpAny

        l1smooth = gaussian_filter1d(l1, 300)
        l2smooth = gaussian_filter1d(l2, 300)
        l3smooth = gaussian_filter1d(l3, 300)
        # plt.plot(l1)
        # plt.plot(l1smooth)
        # plt.show()
        ll = np.vstack((l1smooth, l2smooth, l3smooth))
        lmin = np.min(ll, axis=0)

        tmpl = np.copy(lmin)
        tmpl[~ q11i] = np.nan
        q11maxi = np.nanargmax(tmpl)
        # print(lmin[q11maxi])
        x11_1 = x1[q11maxi]
        x11_2 = x2[q11maxi]
        x11_3 = x3[q11maxi]
        y11_1 = y1[q11maxi]
        y11_2 = y2[q11maxi]
        y11_3 = y3[q11maxi]

        tmpl = np.copy(lmin)
        tmpl[~ q12i] = np.nan
        q12maxi = np.nanargmax(tmpl)
        # print(lmin[q12maxi])
        x12_1 = x1[q12maxi]
        x12_2 = x2[q12maxi]
        x12_3 = x3[q12maxi]
        y12_1 = y1[q12maxi]
        y12_2 = y2[q12maxi]
        y12_3 = y3[q12maxi]

        tmpl = np.copy(lmin)
        tmpl[~ q21i] = np.nan
        q21maxi = np.nanargmax(tmpl)
        # print(lmin[q21maxi])
        x21_1 = x1[q21maxi]
        x21_2 = x2[q21maxi]
        x21_3 = x3[q21maxi]
        y21_1 = y1[q21maxi]
        y21_2 = y2[q21maxi]
        y21_3 = y3[q21maxi]

        tmpl = np.copy(lmin)
        tmpl[~ q22i] = np.nan
        q22maxi = np.nanargmax(tmpl)
        # print(lmin[q22maxi])
        x22_1 = x1[q22maxi]
        x22_2 = x2[q22maxi]
        x22_3 = x3[q22maxi]
        y22_1 = y1[q22maxi]
        y22_2 = y2[q22maxi]
        y22_3 = y3[q22maxi]

        cam1Pts = np.array([[x11_1, y11_1], [x12_1, y12_1], [x21_1, y21_1],
                            [x22_1, y22_1]]).astype(np.float32)
        cam2Pts = np.array([[x11_2, y11_2], [x12_2, y12_2], [x21_2, y21_2],
                            [x22_2, y22_2]]).astype(np.float32)
        cam3Pts = np.array([[x11_3, y11_3], [x12_3, y12_3], [x21_3, y21_3],
                            [x22_3, y22_3]]).astype(np.float32)

        if showPlots:
            plt.subplot(222)
            plt.scatter(cam1Pts[:, 0], cam1Pts[:, 1])
            plt.scatter(cam2Pts[:, 0], cam2Pts[:, 1])
            plt.scatter(cam3Pts[:, 0], cam3Pts[:, 1])
            plt.plot([mx0, mx3, mx3, mx0, mx0], [my0, my0, my3, my3, my0], 'k')
            plt.plot([mx0, mx3], [my1, my1], 'k')
            plt.plot([mx0, mx3], [my2, my2], 'k')
            plt.plot([mx1, mx1], [my0, my3], 'k')
            plt.plot([mx2, mx2], [my0, my3], 'k')
            plt.legend(["1", "2", "3"])
            plt.title("Reference points")

        transform1 = cv2.getPerspectiveTransform(cam1Pts, cam3Pts)
        transform2 = cv2.getPerspectiveTransform(cam2Pts, cam3Pts)

    coords = np.vstack((x1, y1, np.ones_like(x1)))
    cam1_corrected = np.matmul(transform1, coords)
    cam1_corrected = cam1_corrected / cam1_corrected[-1, :]
    cam1_corrected = cam1_corrected.T

    coords = np.vstack((x2, y2, np.ones_like(x2)))
    cam2_corrected = np.matmul(transform2, coords)
    cam2_corrected = cam2_corrected / cam2_corrected[-1, :]
    cam2_corrected = cam2_corrected.T

    cam3_corrected = np.vstack((x3, y3, np.ones_like(x3))).T

    # cam1_corrected[l1 < 0.999, :] = np.nan
    # cam2_corrected[l2 < 0.999, :] = np.nan
    # cam3_corrected[l3 < 0.999, :] = np.nan

    dThresh = 50

    jumpi1 = np.diff(cam1_corrected, axis=0, prepend=np.zeros((1, 3))) > dThresh
    jumpi1 = np.any(jumpi1, axis=1)
    jumpi1 = binary_dilation(jumpi1, iterations=10)
    l1[jumpi1] = 0.0
    cam1_corrected[jumpi1, :] = np.nan

    jumpi2 = np.diff(cam2_corrected, axis=0, prepend=np.zeros((1, 3))) > dThresh
    jumpi2 = np.any(jumpi2, axis=1)
    jumpi2 = binary_dilation(jumpi2, iterations=10)
    l2[jumpi2] = 0.0
    cam2_corrected[jumpi2, :] = np.nan

    jumpi3 = np.diff(cam3_corrected, axis=0, prepend=np.zeros((1, 3))) > dThresh
    jumpi3 = np.any(jumpi3, axis=1)
    jumpi3 = binary_dilation(jumpi3, iterations=10)
    l3[jumpi3] = 0.0
    cam3_corrected[jumpi3, :] = np.nan

    lfac1 = np.interp(l1, [0.999, 1], [0.0, 1.0])
    lfac2 = np.interp(l2, [0.999, 1], [0.0, 1.0])
    lfac3 = np.interp(l3, [0.999, 1], [0.0, 1.0])
    ltot = lfac1 + lfac2 + lfac3
    combo = (cam1_corrected * np.tile(lfac1, (3, 1)).T +
             cam2_corrected * np.tile(lfac2, (3, 1)).T +
             cam3_corrected * np.tile(lfac3, (3, 1)).T)
    combo = combo / np.tile(ltot, (3, 1)).T

    lthresh = 0.5
    combo[ltot < lthresh, :] = np.nan

    if showPlots:
        p1 = cam1_corrected[:, :]
        p1[~ (lfac1 > 0), :] = np.nan
        p2 = cam2_corrected[:, :]
        p2[~ (lfac2 > 0), :] = np.nan
        p3 = cam3_corrected[:, :]
        p3[~ (lfac3 > 0), :] = np.nan
        plt.subplot(223)
        plt.plot(p1[:, 0], p1[:, 1])
        plt.plot(p2[:, 0], p2[:, 1])
        plt.plot(p3[:, 0], p3[:, 1])
        plt.title("Corrected")
        # plt.legend(["1", "2", "3"])

        plt.subplot(224)
        plt.plot(combo[:, 0], combo[:, 1])
        plt.title("Combination")
        plt.show()

    smoothAmt = 3
    smoothCombo = np.copy(combo)
    x = smoothCombo[:, 0]
    y = smoothCombo[:, 1]
    xidx = np.arange(len(x))
    notNanPos = ~ np.isnan(smoothCombo[:, 0])
    x = np.interp(xidx, xidx[notNanPos], x[notNanPos])
    y = np.interp(xidx, xidx[notNanPos], y[notNanPos])
    x = gaussian_filter1d(x, smoothAmt)
    y = gaussian_filter1d(y, smoothAmt)

    if showPlots:
        plt.plot(x, y)
        plt.show()

    return x, y, transform1, transform2


def loadDLCCoords(dlcH5FileName, plot=False):
    print(dlcH5FileName)
    dat = pd.read_hdf(dlcH5FileName)
    # print(dat)

    if plot:
        dat.plot()
        plt.show()

    return dat


def makeLabeledVideo(sessionName, phaseName, x, y):
    inVidFileName = f"/media/WDC6/DLC/trainingVideos/{sessionName}_{phaseName}_cam3.mp4"
    inVid = cv2.VideoCapture(inVidFileName)
    outVidFileName = f"/media/WDC6/DLC/trainingVideos/{sessionName}_{phaseName}_combined_label.mp4"
    outVid = cv2.VideoWriter(outVidFileName, cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (344, 256))

    ii = 0
    w = 5
    while True:
        ret, frame = inVid.read()
        if not ret:
            break

        if (not np.isnan(x[ii])) and (not np.isnan(y[ii])):
            x0 = max(0, int(x[ii]) - w)
            x1 = min(343, int(x[ii]) + w)
            y0 = max(0, int(y[ii]) - w)
            y1 = min(255, int(y[ii]) + w)
            cv2.line(frame, (x0, y0), (x0, y1), (255, 255, 255), 1)
            cv2.line(frame, (x0, y0), (x1, y0), (255, 255, 255), 1)
            cv2.line(frame, (x1, y1), (x0, y1), (255, 255, 255), 1)
            cv2.line(frame, (x1, y1), (x1, y0), (255, 255, 255), 1)

            # for xx in range(max(0, int(x[ii]) - w), min(343, int(x[ii]) + w)):
            #     for yy in range(max(0, int(y[ii]) - w), min(255, int(y[ii]) + w)):
            #         frame[yy, xx, :] = 255
        outVid.write(frame)
        ii += 1

    inVid.release()
    outVid.release()


def saveCoords(sessionName, phaseName, x, y):
    outFile = f"/media/WDC6/DLC/trainingVideos/{sessionName}_{phaseName}.npz"
    tsFile = f"/media/WDC6/DLC/trainingVideos/{sessionName}_{phaseName}_timestamps.npz"
    ts = np.load(tsFile)["timestamps"]
    np.savez(outFile, x1=x, y1=y, timestamp=ts)


def processSessionPhase(sessionName, phaseName, t1, t2):
    dlcH5File = \
        f"/media/WDC6/DLC/trainingVideos/{sessionName}_{phaseName}_cam1DLC_resnet50_Camera_cam1Sep14shuffle1_50005.h5"
    cam1Coords = loadDLCCoords(dlcH5File)
    dlcH5File = \
        f"/media/WDC6/DLC/trainingVideos/{sessionName}_{phaseName}_cam2DLC_resnet50_Camera_cam2Sep14shuffle1_50005.h5"
    cam2Coords = loadDLCCoords(dlcH5File)
    dlcH5File = \
        f"/media/WDC6/DLC/trainingVideos/{sessionName}_{phaseName}_cam3DLC_resnet50_Camera_cam3Sep14shuffle1_50005.h5"
    cam3Coords = loadDLCCoords(dlcH5File)
    x, y, t1, t2 = convertDLCToTrodesCoords(cam1Coords, cam2Coords, cam3Coords, sessionName, t1, t2)
    makeLabeledVideo(sessionName, phaseName, x, y)
    exit()
    saveCoords(sessionName, phaseName, x, y)
    return t1, t2


def processSession(sessionName, t1, t2):
    print("==========================")
    print(sessionName)
    print("==========================")
    print("============")
    print("Task")
    print("============")
    t1, t2 = processSessionPhase(sessionName, "task", t1, t2)
    print("============")
    print("Probe")
    print("============")
    processSessionPhase(sessionName, "probe", t1, t2)
    return t1, t2


if __name__ == "__main__":
    allProcessedSessions = glob("/media/WDC6/DLC/trainingVideos/*task*_Camera_cam1*.h5")
    print(len(allProcessedSessions))
    allProcessedSessions = [
        s for s in allProcessedSessions if os.path.exists(s.replace("task", "probe"))]
    print(len(allProcessedSessions))
    allProcessedSessions = ["_".join(s.split("/")[-1].split("_")[0:2])
                            for s in allProcessedSessions]
    print(allProcessedSessions)
    t1 = None
    t2 = None
    for sessionName in allProcessedSessions:
        t1, t2 = processSession(sessionName, t1, t2)
