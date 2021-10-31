import os

# data_dir = "/media/WDC7/B12/bradtasksessions/"
notes_dir = "/home/wcroughan/repos/labrepos/labnotes/"
filename = os.path.join(notes_dir, "B12.txt")
minimum_date = "20210916"
excluded_dates = ["20210922", "20211011", "20211013", "20211014"]
out_dir = "/media/WDC7/B12/bradtasksessions/behavior_notes"

with open(filename) as f:
    datestr = "20210801"
    saved = False
    homewell = None
    aways = None
    lastaway = None
    condition = None
    lastwell = None
    runinday = 1
    for l in f.readlines():
        l = l.strip()

        if l.startswith("2021-"):
            ds = l.split('-')
            datestr = "{}{}{}".format(ds[0], ds[1], ds[2])
            runinday = 1
            continue

        if datestr < minimum_date or datestr in excluded_dates:
            continue

        if len(l) == 0:
            # save this record
            if not saved:
                outstr = ""
                outstr += "Condition: {}\n".format(condition)
                outstr += "Thresh: Low\n"
                outstr += "Home: {}\n".format(homewell)
                outstr += "Aways: {}\n".format(' '.join(aways))
                outstr += "Last away: {}\n".format(lastaway)
                outstr += "Last well: {}\n".format(lastwell)
                outstr += "ITI stim on: N\nProbe stim on: N\nRef: 8\nBaseline: 3\n"

                print("{}_{}".format(datestr, runinday))
                print(outstr)
                print("")

                if homewell is None or aways is None or lastaway is None or condition is None or lastwell is None:
                    raise Exception("Missing data for date {}_{}".format(datestr, runinday))

                with open(os.path.join(out_dir, "{}_{}.txt".format(datestr, runinday)), "w") as fout:
                    fout.write(outstr)

                homewell = None
                aways = None
                lastaway = None
                condition = None
                lastwell = None
                runinday += 1
                saved = True
                continue
        else:
            saved = False

        if ("delay" in l.lower() or "interruption" in l.lower()) and ("baseline" in l or "base" in l) and "ref" in l:
            if "delay" in l.lower():
                condition = "delay"
            else:
                condition = "interruption"

        elif l.startswith("H:"):
            homewell = l.split(' ')[1]

        elif l.startswith("A:"):
            rawAways = l.split(' ')[1:]

        elif l.startswith("F:"):
            aways = l.split(' ')[1:]
            lastaway = aways[-1]
            for w in rawAways:
                if w not in aways:
                    aways.append(w)

        elif "last home" in l.lower():
            lastwell = "Home"

        elif "last well" in l.lower():
            lastwell = "Home" if "home" in l else "Away"

        else:
            # print("Couldn't parse this line: {}".format(l))
            pass
