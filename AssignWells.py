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
if START_WITH_TODAY:
    first_day = datetime.now()
else:
    raise Exception

# skipDays = [0] * NUM_DAYS
skipDays = [0, 0, 0, 0]

output_dir = "./behavior_notes/B13/"
# output_dir = "./"
# output_dir = "/media/fosterlab/WDC4/B8/behavior_notes/"

REST_OF_FILE = "\nThresh: Low\nLast Away: \nLast well: \nITI Stim On: \nProbe Stim On: \nWeight: \n"
PRINT_WELLS = True
SAVE_TO_FILE = True
PRINT_WELL_GRID = True
PRINT_DAY_WELL_GRID = True

all_wells = [i + 1 for i in range(48) if not i % 8 in [0, 7]]
broken_wells = []
# broken_wells = [2, 4, 6, 7, 18, 20, 42, 29, 31, 39, 37, 47, 27]
# broken_wells = [2, 3, 4, 20, 42, 34]
# broken_wells = [34, 15, 21, 22, 28, 29, 44, 38, 35]
# broken_wells = [12, 6, 14]
working_wells = set(all_wells) - set(broken_wells)

def quadrant(well):
    if i < 25:
        if i % 8 in [2, 3, 4]:
            return 1
        else:
            return 2
    else:
        if i % 8 in [2, 3, 4]:
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
    wellgrid = np.array(['.'] * 36).reshape(6,6)
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

    NUM_AWAY_WELLS_PER_SESSION = 9

    if NUM_DAYS * RUNS_PER_DAY > len(OFF_WALL_WELLS):
        raise Exception("UNIMPLEMENTED: too many days")

    if ENFORCE_HOME_OFF_WALL:
        # home_wells = rd.sample(OFF_WALL_WELLS, NUM_DAYS * RUNS_PER_DAY)
        # while not approveHomeWells(home_wells):
            # home_wells = rd.sample(OFF_WALL_WELLS, NUM_DAYS * RUNS_PER_DAY)

        n = math.ceil(NUM_DAYS * RUNS_PER_DAY / 4)
        home_wells = rd.sample(OFFWALLQ1, n) + rd.sample(OFFWALLQ2, n) + rd.sample(OFFWALLQ3, n) + rd.sample(OFFWALLQ4, n)  
        rd.shuffle(home_wells)
        home_wells = home_wells[0:NUM_DAYS * RUNS_PER_DAY]

    else:
        home_wells = rd.sample(working_wells, NUM_DAYS * RUNS_PER_DAY)

    day_condition_order = [0, 1, 2, 3]
    for di in range(NUM_DAYS):
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
            else:
                condition = "Delayed"
            outstr1 = "Home: " + wells[0] + "\nAways: " + " ".join(wells[1:])
            if PRINT_WELLS:
                print(outstr1)
                print(condition)

            if PRINT_WELL_GRID:
                print(wellGridString(wells[0], wells[1:]))

            if PRINT_DAY_WELL_GRID:
                daywells = daywells | set(wells)

            if SAVE_TO_FILE:
                outstr = outstr1 + "\nCondition: " + condition + REST_OF_FILE
                fname = os.path.join(output_dir, thisday.strftime(
                    "%Y%m%d_{}.txt".format(ri + 1)))
                if os.path.exists(fname):
                    con = "a"
                    while not (con in ["y", "n"]):
                        con = input("File {} exists, overwrite? (y/n):".format(fname))

                    if con == "n":
                        print("aborting")
                        exit()

                with open(fname, "w") as f:
                    f.write(outstr)
                    f.write("\n\n" + wellGridString(None, daywells) + "\n")

        if PRINT_DAY_WELL_GRID:
            print("DAY GRID: {}".format(thisday.strftime("%Y%m%d")))
            print(wellGridString(None, daywells))

        thisday = thisday + timedelta(days=1)
        thisday = thisday + timedelta(days=skipDays[di])
