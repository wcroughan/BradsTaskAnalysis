class BTSession:
    """
    Contains all data for a session on Brad's task with probe
    Also has references to previous and next session.
    """

    def __init__(self):
        # ==================================
        # Info about the session
        # ==================================
        # The previous session chronologically, if one exists
        self.prevSession = None
        # The next session chronologically, if one exists
        self.nextSession = None
        # date object representing the day of this recording
        self.date = None

        # Just the date part of session filename (i.e. "20200112")
        self.date_str = ""
        # Data string in general. May modify in future by appending "S1" for first session if doing multiple in one day
        self.name = ""
        # name of raw data folder in which brad's task part of session was recorded
        self.bt_dir = ""
        # name of raw data folder in which ITI part of session was recorded. May be missing (empty string). May be same as bt_dir
        self.iti_dir = ""
        # name of raw data folder in which probe part of session was recorded. May be missing (empty string). May be same as bt_dir
        self.probe_dir = ""

        # Some flags indicated whether ITI was recorded and whether ITI and probe are in the same rec file or not
        self.separate_iti_file = False
        self.recorded_iti = False
        self.separate_probe_file = False

        # more flags from info file
        self.ripple_detection_threshold = 0.0
        self.last_away_well = 0
        self.ended_on_home = False
        self.ITI_stim_on = False
        self.probe_stim_on = False

        # Any other notes that are stored in the info file are added here. Each list entry is one line from that file
        self.notes = []

        # position coordinates of home well
        self.home_x = 0
        self.home_y = 0
        self.away_xs = []
        self.away_ys = []

        # Well number of home well
        self.home_well = 0
        # Well number of away wells
        self.away_wells = []
        self.num_away_found = 0

        # Flags indicating stim condition
        self.isRippleInterruption = False
        self.isDelayedInterruption = False
        self.isNoInterruption = False
        self.ripple_detection_tetrodes = []

        # Rat weight if it was recorded
        self.rat_weight = 0

        # ==================================
        # Raw data
        # ==================================
        # Position data during brad's task
        self.bt_pos_ts = []
        self.bt_pos_xs = []
        self.bt_pos_ys = []

        # Position data during probe
        self.probe_pos_ts = []
        self.probe_pos_xs = []
        self.probe_pos_ys = []

        # LFP data is huge, so only load on demand
        # brad's task
        self.bt_lfp_fnames = []
        self.bt_lfp_start_ts = 0
        self.bt_lfp_end_ts = 0
        self.bt_lfp_start_idx = 0
        self.bt_lfp_end_idx = 0

        # ITI
        self.iti_lfp_fnames = []
        self.iti_lfp_start_ts = 0
        self.iti_lfp_end_ts = 0
        self.iti_lfp_start_idx = 0
        self.iti_lfp_end_idx = 0

        # probe
        self.probe_lfp_fnames = []
        self.probe_lfp_start_ts = 0
        self.probe_lfp_end_ts = 0
        self.probe_lfp_start_idx = 0
        self.probe_lfp_end_idx = 0

        # ==================================
        # Analyzed data: Brad's task
        # ==================================
        self.home_well_find_times = []
        self.home_well_leave_times = []
        self.home_well_latencies = []
        self.home_well_displacements = []
        self.home_well_distances = []

        self.away_well_find_times = []
        self.away_well_leave_times = []
        self.away_well_latencies = []
        self.away_well_displacements = []
        self.away_well_distances = []
        self.visited_away_wells = []

        # ==================================
        # Analyzed data: Probe
        # ==================================
        # self.home_well_entry_times = []
        # self.mean_dist_to_home_well = []

        # # control home well, rotate home well 180 degrees around middle of environment
        # self.ctrl_home_well = 0
        # self.ctrl_home_x = 0
        # self.ctrl_home_y = 0
        # self.ctrl_home_well_entry_times = []
        # self.mean_dist_to_ctrl_home_well = []

        # self.prev_home_well_entry_times = []
        # self.mean_dist_to_prev_home_well = []

        # self.away_well_entry_times = []
        # self.mean_dist_to_away_wells = []

        # analyzing paths, separating by velocity
        self.bt_vel_cm_s = []
        self.bt_is_mv = []
        self.bt_mv_xs = []
        self.bt_still_xs = []
        self.bt_mv_ys = []
        self.bt_still_ys = []
        self.probe_vel_cm_s = []
        self.probe_is_mv = []
        self.probe_mv_xs = []
        self.probe_still_xs = []
        self.probe_mv_ys = []
        self.probe_still_ys = []

        # avg dist to home and times at which rat entered home region
        self.probe_nearest_wells = []
        self.probe_well_entry_idxs = []
        self.probe_well_exit_idxs = []
        self.probe_well_entry_times = []
        self.probe_well_exit_times = []
        self.probe_home_well_entry_times = []
        self.probe_mean_dist_to_home_well = []
        self.probe_mv_mean_dist_to_home_well = []
        self.probe_still_mean_dist_to_home_well = []
        self.probe_ctrl_home_well_entry_times = []
        self.probe_mean_dist_to_ctrl_home_well = []
        self.probe_mv_mean_dist_to_ctrl_home_well = []
        self.probe_still_mean_dist_to_ctrl_home_well = []

        self.bt_nearest_wells = []
        self.bt_well_entry_idxs = []
        self.bt_well_exit_idxs = []
        self.bt_well_entry_times = []
        self.bt_well_exit_times = []
        self.bt_home_well_entry_times = []
        self.bt_mean_dist_to_home_well = []
        self.bt_mv_mean_dist_to_home_well = []
        self.bt_still_mean_dist_to_home_well = []
        self.bt_ctrl_home_well_entry_times = []
        self.bt_mean_dist_to_ctrl_home_well = []
        self.bt_mv_mean_dist_to_ctrl_home_well = []
        self.bt_still_mean_dist_to_ctrl_home_well = []
