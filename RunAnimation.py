from BTData import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from itertools import groupby
import random
from scipy import stats, signal
import scipy

data_filename = "/media/WDC4/martindata/bradtask/martin_bradtask.dat"
alldata = BTData()
alldata.loadFromFile(data_filename)
all_sessions = alldata.getSessions()

output_dir = '/media/WDC4/martindata/processed_data/animations'

# SHOW_OUTPUT_PLOTS = True
SAVE_OUTPUT_PLOTS = True
TRODES_SAMPLING_RATE = 30000


class RunAnimator:
    def __init__(self, out_fname, xdata, ydata, side_dat1=None, side_dat2=None, cdata=None,
                 side_dat_name1="", side_dat_name2="", xlim=(0, 1200), ylim=(0, 1000),
                 PLOT_BEHAVIOR_FRAMES=24, frame_len=30, saveVideo=False, additional_artist_builder=None):
        self._xdata = np.copy(xdata)
        self._ydata = np.copy(ydata)
        if cdata is None:
            self.has_cdata = False
        else:
            self.has_cdata = True
            self._cdata = np.copy(cdata).astype('float64')
            self._cdata -= np.nanmin(self._cdata)
            self._cdata *= 1.0 / np.nanmax(self._cdata)

        self._xlim = xlim
        self._ylim = ylim

        self._frame_len = frame_len
        # if saveVideo:
        # self._frame_len /= 8

        self._outfname = out_fname + ".mp4"
        self._saveVideo = saveVideo

        plt.rcParams['keymap.xscale'].remove('k')
        plt.rcParams['keymap.yscale'].remove('l')

        if side_dat1 is not None:
            self.has_side_data = True
            self._side_dat1 = np.copy(side_dat1)
            self._side_dat2 = np.copy(side_dat2)
            figsz = [13, 4.8]
            self.fig = plt.figure(0, figsize=figsz)
            self.ax1 = plt.subplot(1, 2, 1)
            self.ax2 = plt.subplot(1, 2, 2)
            self.line, = self.ax1.plot([], [], lw=2)
            self.ax1.grid()
            self.ax2.set_ylabel(side_dat_name2)
            self.ax2.set_xlabel(side_dat_name1)
            self.ax1.set_yscale('linear')
            self.ax2.set_yscale('linear')

            self.sideline_all, = self.ax2.plot(side_dat1, side_dat2, lw=0.5, c='grey')
            self.sideline, = self.ax2.plot([], [], lw=2, c='red')
        else:
            self.has_side_data = False
            figsz = [6.4, 4.8]
            self.fig = plt.figure(0, figsize=figsz)
            self.ax1 = plt.subplot(1, 1, 1)
            self.line, = self.ax1.plot([], [], lw=2)
            self.ax1.grid()
            self.ax1.set_yscale('linear')

        self.beh_frames_to_plot = PLOT_BEHAVIOR_FRAMES
        self.cmap = plt.cm.get_cmap('coolwarm')
        # self.cmap = plt.cm.get_cmap('nipy_spectral')

        self._max_frame = self._xdata.size
        self._delta_frame = 1
        self._pause_delta_frame = 0
        self._play_delta_frame = 1
        self._frame_jump_dist = 30
        self._is_paused = False

        if additional_artist_builder is None:
            self.has_addnl_artists = False
        else:
            self.has_addnl_artists = True
            self._additional_artists = additional_artist_builder(self.ax1)

    def run_animation(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.ani = anim.FuncAnimation(self.fig, lambda i: self.animate(i), frames=self.update_time,
                                      repeat=False, init_func=self.init_plot, interval=self._frame_len,
                                      save_count=self._max_frame)

        # plt.show()

        # Writer = anim.writers['ffmpeg']
        # writer = Writer(fps=int((1000.0/float(self._frame_len))))
        # writer = Writer()
        # self.ani.save(self._outfname, writer=writer)
        if self._saveVideo:
            self.ani.save(self._outfname, 'ffmpeg')
        else:
            plt.show()

    def update_time(self):
        self._frame = 0
        while self._frame < self._max_frame:
            yield self._frame
            self._frame += self._delta_frame
            if self._frame < 0:
                self._frame = 0

    def on_press(self, event):
        if event.key.isspace() or event.key == 'k':
            if self._is_paused:
                self._is_paused = False
                self._delta_frame = self._play_delta_frame
                # self.ani.event_source.start()
            else:
                self._is_paused = True
                self._delta_frame = self._pause_delta_frame
                # self.ani.event_source.stop()
        elif event.key == 'j':
            self._play_delta_frame = -abs(self._play_delta_frame)
            if not self._is_paused:
                self._delta_frame = self._play_delta_frame
        elif event.key == 'l':
            self._play_delta_frame = abs(self._play_delta_frame)
            if not self._is_paused:
                self._delta_frame = self._play_delta_frame
        # elif event.key == 'n':
        #     self._frame = min(self._max_frame, self._frame + self._frame_jump_dist)
        # elif event.key == 'p':
        #     self._frame = max(0, self._frame - self._frame_jump_dist)
        elif event.key == 'i':
            self._play_delta_frame *= 2
            if not self._is_paused:
                self._delta_frame = self._play_delta_frame
        elif event.key == 'u':
            if abs(self._play_delta_frame) > 1:
                self._play_delta_frame //= 2
                if not self._is_paused:
                    self._delta_frame = self._play_delta_frame

    def init_plot(self):
        self.ax1.set_xlim(self._xlim[0], self._xlim[1])
        self.ax1.set_ylim(self._ylim[0], self._ylim[1])
        self.ax1.set_yscale('linear')
        self.line.set_data([], [])
        if self.has_addnl_artists:
            self._additional_artists.init_plot()
        if self.has_side_data:
            self.sideline.set_data([], [])
            self.ax2.set_yscale('linear')
            return self.line, self.sideline
        else:
            return self.line,

    def animate(self, i):
        i1 = max(0, i - self.beh_frames_to_plot)
        self.line.set_data(self._xdata[i1:i], self._ydata[i1:i])
        if self.has_cdata:
            self.line.set_color(self.cmap(self._cdata[i]))
        res = [self.line]

        if self.has_side_data:
            self.sideline.set_data(self._side_dat1[i1:i], self._side_dat2[i1:i])
            res.append(self.sideline)

        if self.has_addnl_artists:
            self._additional_artists.animate(i)
            res += self._additional_artists.artist_list()

        return tuple(res)


class CurvatureArtist:
    def __init__(self, sesh, ax):
        self._ax = ax
        self.radius = 8 * 5
        # self.circle = plt.Circle((0, 0), radius=self.radius, fill=False)
        self.vec1 = ax.plot([], [], lw=2)[0]
        self.vec2 = ax.plot([], [], lw=2)[0]
        self.subpath = ax.plot([], [], lw=2)[0]

        self.xdat = np.array(sesh.probe_pos_xs)
        self.ydat = np.array(sesh.probe_pos_ys)
        self.fullpath = ax.plot(self.xdat, self.ydat, lw=1, c=(0.5, 0.5, 0.5, 0.5), zorder=0)

        self.path_i1 = sesh.probe_curvature_i1
        self.path_i2 = sesh.probe_curvature_i2 + 1
        self.dxf = sesh.probe_curvature_dxf
        self.dyf = sesh.probe_curvature_dyf
        self.dxb = sesh.probe_curvature_dxb
        self.dyb = sesh.probe_curvature_dyb

        self._cdata = np.copy(sesh.probe_curvature).astype('float64')
        self._cdata -= np.nanmin(self._cdata)
        self._cdata *= 1.0 / np.nanmax(self._cdata)
        self.cmap = plt.cm.get_cmap('coolwarm')

    def init_plot(self):
        # self.circle.set_center((self.xdat[0], self.ydat[0]))
        self.vec1.set_data([], [])
        self.vec2.set_data([], [])
        self.subpath.set_data([], [])

    def animate(self, i):
        # self.circle.set_center((self.xdat[i], self.ydat[i]))
        self.vec1.set_data([self.xdat[i], self.xdat[i]+self.dxf[i]],
                           [self.ydat[i], self.ydat[i]+self.dyf[i]])
        self.vec2.set_data([self.xdat[i], self.xdat[i]-self.dxb[i]],
                           [self.ydat[i], self.ydat[i]-self.dyb[i]])
        if (not (np.isnan(self.path_i1[i]) or np.isnan(self.path_i2[i]))):
            self.subpath.set_data(self.xdat[int(self.path_i1[i]):int(self.path_i2[i])],
                                  self.ydat[int(self.path_i1[i]):int(self.path_i2[i])])
            self.subpath.set_color(self.cmap(self._cdata[i]))
        else:
            self.subpath.set_data([], [])

    def artist_list(self):
        # return [self.circle, self.vec1, self.vec2]
        return [self.vec1, self.vec2, self.subpath]


random.shuffle(all_sessions)
for sesh in all_sessions:
    print(sesh.name)

    POS_FRAME_RATE = scipy.stats.mode(
        np.diff(sesh.probe_pos_ts))[0] / float(TRODES_SAMPLING_RATE)

    if False:
        d1 = (sesh.bt_vel_cm_s.size - sesh.bt_ball_displacement.size) // 2 + 1
        ball = sesh.bt_ballisticity[0:sesh.bt_sm_vel[d1:-d1].size]

        ball_sigma_sec = 0.3
        ball_sigma_frames = ball_sigma_sec / POS_FRAME_RATE[0]
        sm_ball = scipy.ndimage.gaussian_filter1d(
            ball, ball_sigma_frames)
        # sm_ball = ball

        # sm_ball -= np.min(sm_ball)
        # sm_ball *= 1.0/np.max(sm_ball)
        # sm_ball = 1.0 / sm_ball
        animator = RunAnimator(sesh.name,
                               sesh.bt_pos_xs[d1:-d1],
                               sesh.bt_pos_ys[d1:-d1],
                               sesh.bt_sm_vel[d1:-d1],
                               sm_ball,
                               cdata=sm_ball,
                               side_dat_name1="Smoothed Velocity",
                               side_dat_name2="Ballisticity",
                               frame_len=POS_FRAME_RATE[0] * 1000.0)
        animator.run_animation()

    if True:
        USE_BOUTS_AS_COLORS = False
        USE_SIDE_PLOTS = False
        SHOW_CURVE_VECS = True
        if USE_BOUTS_AS_COLORS:
            cdat = sesh.probe_bout_category
        else:
            cdat = sesh.probe_curvature
            # cdat = sesh.probe_ballisticity
            # cdat = np.pad(cdat, (sesh.probe_curvature.size - cdat.size, 0))

            cdat_sigma_sec = 0.3
            cdat_sigma_frames = cdat_sigma_sec / POS_FRAME_RATE[0]
            sm_cdat = scipy.ndimage.gaussian_filter1d(
                cdat, cdat_sigma_frames)

        if USE_SIDE_PLOTS:
            dat1 = sesh.probe_curvature
            dat2 = sesh.probe_ballisticity

            dd = abs(dat1.size - dat2.size)

            if dat1.size < dat2.size:
                dat1 = np.pad(dat1, (int(np.floor(float(dd)/2.0)), int(np.ceil(float(dd)/2.0))))
            elif dd > 0:
                dat2 = np.pad(dat2, (int(np.floor(float(dd)/2.0)), int(np.ceil(float(dd)/2.0))))

            dat1_sigma_sec = 0.3
            dat1_sigma_frames = dat1_sigma_sec / POS_FRAME_RATE[0]
            sm_dat1 = scipy.ndimage.gaussian_filter1d(
                dat1, dat1_sigma_frames)

            dat2_sigma_sec = 0.3
            dat2_sigma_frames = dat1_sigma_sec / POS_FRAME_RATE[0]
            sm_dat2 = scipy.ndimage.gaussian_filter1d(
                dat2, dat2_sigma_frames)

            if dat1.size < cdat.size:
                dd = cdat.size - dat1.size
                dat1 = np.pad(dat1, (int(np.floor(float(dd)/2.0)), int(np.ceil(float(dd)/2.0))))
                dat2 = np.pad(dat2, (int(np.floor(float(dd)/2.0)), int(np.ceil(float(dd)/2.0))))

            animator = RunAnimator(sesh.name + "_curve_and_ballisticity",
                                   sesh.probe_pos_xs[0:-1],
                                   sesh.probe_pos_ys[0:-1],
                                   cdata=cdat,
                                   side_dat1=sm_dat1,
                                   side_dat2=sm_dat2,
                                   side_dat_name1="Curvature",
                                   side_dat_name2="Ballisticity",
                                   frame_len=POS_FRAME_RATE[0] * 1000.0,
                                   saveVideo=True if SAVE_OUTPUT_PLOTS else False)

        elif SHOW_CURVE_VECS:
            animator = RunAnimator(sesh.name + "_curvature",
                                   sesh.probe_pos_xs[0:-1],
                                   sesh.probe_pos_ys[0:-1],
                                   cdata=cdat,
                                   frame_len=POS_FRAME_RATE[0] * 1000.0,
                                   saveVideo=True if SAVE_OUTPUT_PLOTS else False,
                                   PLOT_BEHAVIOR_FRAMES=2,
                                   additional_artist_builder=lambda ax: CurvatureArtist(sesh, ax))
        else:
            animator = RunAnimator(sesh.name + "_cbouts",
                                   sesh.probe_pos_xs[0:-1],
                                   sesh.probe_pos_ys[0:-1],
                                   cdata=cdat,
                                   frame_len=POS_FRAME_RATE[0] * 1000.0,
                                   saveVideo=True if SAVE_OUTPUT_PLOTS else False)

        animator.run_animation()

        break
