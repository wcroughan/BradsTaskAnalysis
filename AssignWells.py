from datetime import datetime, timedelta
import random as rd
import os

RUNS_PER_DAY = 1
NUM_DAYS = 1
START_WITH_TODAY = True
if START_WITH_TODAY:
    first_day = datetime.now()
else:
    raise Exception
output_dir = "./"

REST_OF_FILE = "\nThresh: Low\nLast Away: \nLast well: \nITI Stim On: \nProbe Stim On: \n"

all_wells = [i + 1 for i in range(48) if not i % 8 in [0, 7]]
broken_wells = [2, 4, 6, 7, 18]
working_wells = set(all_wells) - set(broken_wells)

thisday = first_day

day_condition_order = [0, 1, 2, 3]
for di in range(NUM_DAYS):
    if di % 4 == 0:
        rd.shuffle(day_condition_order)
    dc = day_condition_order[di % 4]
    this_day_condition_order = [not (dc & 1), not (dc & 2)]

    for ri in range(RUNS_PER_DAY):
        wells = list(map(lambda i: str(i), rd.sample(working_wells, 10)))
        if this_day_condition_order[ri]:
            condition = "Interruption"
        else:
            condition = "Delayed"
        outstr = "Home: " + wells[0] + "\nAways: " + \
            " ".join(wells[1:]) + "\nCondition: " + condition + REST_OF_FILE

        fname = os.path.join(output_dir, thisday.strftime("%Y%m%d_{}.txt".format(ri + 1)))
        if os.path.exists(fname):
            con = "a"
            while not (con in ["y", "n"]):
                con = input("File {} exists, overwrite? (y/n):".format(fname))

            if con == "n":
                print("aborting")
                exit()

        with open(fname, "w") as f:
            f.write(outstr)

    thisday = thisday + timedelta(days=1)
