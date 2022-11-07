from datetime import datetime, timedelta
import random as rd
import os
import numpy as np
import math

# 0 - Brad's task
# 1 - choose one well from each quadrant
TASK_VERSION = 0

RUNS_PER_DAY = 2
NUM_DAYS = 4
START_WITH_TODAY = True
OVERWRITE_FILES_NO_CONFIRM = False
if START_WITH_TODAY:
    first_day = datetime.now()
else:
    first_day = datetime.now() + timedelta(days=-1)

# skipDays = [0] * NUM_DAYS
skipDays = [0, 0, 0, 0]

output_dir = os.path.join(os.getcwd(), "behavior_notes", "B16-20")
# output_dir = "./"
# output_dir = "/media/fosterlab/WDC4/B8/behavior_notes/"
ratNames = ["B16", "B17", "B18", "B19", "B20"]

REST_OF_FILE = "\nThresh: Low\nLast Away: \nLast well: \nITI Stim On: \nProbe Stim On: \nWeight: \n"
PRINT_WELLS = True
PRINT_PROBE_FILL_TIME = True
PRINT_RAT_ORDER = True
SAVE_TO_FILE = True
PRINT_WELL_GRID = False
PRINT_DAY_WELL_GRID = True

all_wells = [i + 1 for i in range(48) if not i % 8 in [0, 7]]
# broken_wells = []
broken_wells = [46]
working_wells = set(all_wells) - set(broken_wells)

probeFillTimes = np.linspace(60, 60*4, 4)


def quadrant(well):
    if well < 25:
        if well % 8 in [2, 3, 4]:
            return 1
        else:
            return 2
    else:
        if well % 8 in [2, 3, 4]:
            return 3
        else:
            return 4


ENFORCE_HOME_OFF_WALL = True
WALL_WELLS = [str(i) for i in all_wells if i < 9 or i > 40 or i % 8 in [2, 7]]
OFF_WALL_WELLS = [w for w in working_wells if not (str(w) in WALL_WELLS)]
q1 = [i for i in working_wells if i % 8 in [2, 3, 4] and i < 25]
q2 = [i for i in working_wells if i % 8 in [5, 6, 7] and i < 25]
q3 = [i for i in working_wells if i % 8 in [2, 3, 4] and i > 25]
q4 = [i for i in working_wells if i % 8 in [5, 6, 7] and i > 25]
OFFWALLQ1 = [w for w in q1 if w in OFF_WALL_WELLS]
OFFWALLQ2 = [w for w in q2 if w in OFF_WALL_WELLS]
OFFWALLQ3 = [w for w in q3 if w in OFF_WALL_WELLS]
OFFWALLQ4 = [w for w in q4 if w in OFF_WALL_WELLS]


def wellGridString(homewell, wellnums):
    wellgrid = np.array(['.'] * 36).reshape(6, 6)
    for wn in wellnums:
        wellgrid[int(wn) // 8, (int(wn)-2) % 8] = "a"
    if homewell is not None:
        wellgrid[int(homewell) // 8, (int(homewell)-2) % 8] = "h"
    return '\n'.join(reversed([''.join(row) for row in wellgrid]))


if TASK_VERSION == 1:
    print("Wells: {}, {}, {}, {}".format(
        rd.sample(q1, 1)[0],
        rd.sample(q2, 1)[0],
        rd.sample(q3, 1)[0],
        rd.sample(q4, 1)[0]
    ))
    exit()


def approveHomeWells(home_wells):
    # if quadrant(home_wells[0]) != 1 or quadrant(home_wells[1]) != 4:
    #     return False
    i = 0
    while i < len(home_wells) - 1:
        if quadrant(home_wells[i]) == quadrant(home_wells[i+1]):
            return False
        i += 2

    return any([w in q1 for w in home_wells]) and \
        any([w in q2 for w in home_wells]) and \
        any([w in q3 for w in home_wells]) and \
        any([w in q4 for w in home_wells])


if TASK_VERSION == 0:
    thisday = first_day
    ratOrder = [v for v in ratNames]

    NUM_AWAY_WELLS_PER_SESSION = 12

    if NUM_DAYS * RUNS_PER_DAY > len(OFF_WALL_WELLS):
        raise Exception("UNIMPLEMENTED: too many days")

    if ENFORCE_HOME_OFF_WALL:
        # home_wells = rd.sample(OFF_WALL_WELLS, NUM_DAYS * RUNS_PER_DAY)
        # while not approveHomeWells(home_wells):
        # home_wells = rd.sample(OFF_WALL_WELLS, NUM_DAYS * RUNS_PER_DAY)

        n = math.ceil(NUM_DAYS * RUNS_PER_DAY / 4)
        home_wells = rd.sample(OFFWALLQ1, n) + rd.sample(OFFWALLQ2, n) + \
            rd.sample(OFFWALLQ3, n) + rd.sample(OFFWALLQ4, n)
        while not approveHomeWells(home_wells):
            rd.shuffle(home_wells)
        home_wells = home_wells[0:NUM_DAYS * RUNS_PER_DAY]

    else:
        home_wells = rd.sample(working_wells, NUM_DAYS * RUNS_PER_DAY)

    day_condition_order = [0, 1, 2, 3]
    for di in range(NUM_DAYS):
        if di % 4 == 0:
            ctrlFillTimeIdx = 0
            swrFillTimeIdx = 0
            ctrlFillTimes = np.copy(probeFillTimes)
            swrFillTimes = np.copy(probeFillTimes)
            rd.shuffle(ctrlFillTimes)
            rd.shuffle(swrFillTimes)

        if di % 4 in [0, 2]:
            tf = rd.random() < 0.5
            this_day_condition_order = [tf]
            this_day_condition_order.append(tf if di == 2 else not tf)
        else:
            this_day_condition_order = [not tf for tf in this_day_condition_order]

        # if di % 4 == 0:
        #     rd.shuffle(day_condition_order)
        # dc = day_condition_order[di % 4]
        # this_day_condition_order = [not (dc & 1), not (dc & 2)]

        rd.shuffle(ratOrder)

        daywells = set()

        for ri in range(RUNS_PER_DAY):
            hi = di * RUNS_PER_DAY + ri
            home_well = str(home_wells[hi])
            wells = [home_well]
            while home_well in wells:
                wells = list(map(lambda i: str(i), rd.sample(
                    working_wells, NUM_AWAY_WELLS_PER_SESSION + 1)))
            wells[0] = home_well

            # if ENFORCE_HOME_OFF_WALL:
            #     wells = ["2"]
            #     while wells[0] in WALL_WELLS:
            #         wells = list(map(lambda i: str(i), rd.sample(
            #             working_wells, NUM_AWAY_WELLS_PER_SESSION + 1)))
            # else:
            #     wells = list(map(lambda i: str(i), rd.sample(
            #         working_wells, NUM_AWAY_WELLS_PER_SESSION + 1)))

            # wells = [str(x) for x in [39, 7, 22, 31, 2, 27, 43, 13, 37, 4]]
            if this_day_condition_order[ri]:
                condition = "Interruption"
                probeFillTime = ctrlFillTimes[ctrlFillTimeIdx]
                ctrlFillTimeIdx += 1
            else:
                condition = "Delayed"
                probeFillTime = swrFillTimes[swrFillTimeIdx]
                swrFillTimeIdx += 1

            if PRINT_RAT_ORDER:
                print(ratOrder)

            outstr1 = "Home: " + wells[0] + "\nAways: " + " ".join(wells[1:])
            if PRINT_WELLS:
                print(outstr1)
                print(condition)

            if PRINT_PROBE_FILL_TIME:
                print(f"fill probe at {probeFillTime}s")

            if PRINT_WELL_GRID:
                print(wellGridString(wells[0], wells[1:]))

            # if PRINT_DAY_WELL_GRID:
            daywells = daywells | set(wells)

            if SAVE_TO_FILE:
                outstr = "Order: " + str(ratOrder) + "\n" + outstr1 + "\nCondition: " + condition + \
                    f"\nfill probe at {probeFillTime}s" + REST_OF_FILE
                fname = os.path.join(output_dir, thisday.strftime(
                    "%Y%m%d_{}.txt".format(ri + 1)))
                if os.path.exists(fname) and not OVERWRITE_FILES_NO_CONFIRM:
                    con = "a"
                    while not (con in ["y", "n"]):
                        con = input("File {} exists, overwrite? (y/n):".format(fname))

                    if con == "n":
                        print("aborting")
                        exit()

                with open(fname, "w") as f:
                    f.write(outstr)
                    f.write("\n\n" + wellGridString(None, set(wells)) + "\n")
                    if set(wells) != daywells:
                        f.write("\n\n" + wellGridString(None, set(daywells)) + "\n")
                        for _ in range(3):
                            f.write("\n")

        if PRINT_DAY_WELL_GRID:
            print("DAY GRID: {}".format(thisday.strftime("%Y%m%d")))
            print(wellGridString(None, daywells))

        thisday = thisday + timedelta(days=1)
        thisday = thisday + timedelta(days=skipDays[di])
