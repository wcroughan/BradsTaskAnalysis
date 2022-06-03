import operator
import pandas as pd
import json
import numpy as np
import os
from functools import reduce
from BTSession import BTSession
from BTRestSession import BTRestSession
from datetime import datetime

NP_KEY_PFX = '__numpy_ref__'
NP_LIST_KEY_PFX = '__numpy_list_ref__'


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


class BTData:
    def __init__(self):
        self.filename = ""
        self.allSessions = []
        self.allRestSessions = []

    def loadFromFile(self, filename):
        with open(filename, 'r') as f:
            np_dir = filename + ".numpy_objs"
            self.allSessions = []
            line = f.readline()
            while line:
                if line == "!!Session\n":
                    self.allSessions.append(BTSession())
                    line = f.readline()
                    load_dict = json.loads(line[:-1])

                    self.allSessions[-1].__dict__ = self.processAndLoadDict(load_dict, np_dir)
                    self.allSessions[-1].date = datetime.strptime("{}_{}".format(
                        self.allSessions[-1].date_str, self.allSessions[-1].time_str), "%Y%m%d_%H%M%S")
                elif line == "!!RestSession\n":
                    self.allRestSessions.append(BTRestSession())
                    line = f.readline()
                    load_dict = json.loads(line[:-1])
                    # print(load_dict)

                    self.allRestSessions[-1].__dict__ = self.processAndLoadDict(load_dict, np_dir)
                    self.allRestSessions[-1].date = datetime.strptime("{}_{}".format(
                        self.allRestSessions[-1].date_str, self.allRestSessions[-1].time_str), "%Y%m%d_%H%M%S")

                    # print(self.allRestSessions[-1].restDuration)
                else:
                    print("File parse error!")
                    return -2

                line = f.readline()

            for bi in range(0, len(self.allSessions)):
                if bi > 0:
                    self.allSessions[bi].prevSession = self.allSessions[bi - 1]
                    self.allSessions[bi - 1].nextSession = self.allSessions[bi]

            for rsi, rs in enumerate(self.allRestSessions):
                sn = rs.btwpSessionName
                for bs in self.allSessions:
                    if bs.name == sn:
                        rs.btwpSession = bs
                        break

            self.filename = filename

            return 0
        print("Couldn't open file %s" % filename)
        return -1

    def saveToFile(self, filename):
        with open(filename, 'w') as f:
            np_dir = filename + ".numpy_objs"
            try:
                os.mkdir(np_dir)
            except FileExistsError:
                pass

            for bi, bb in enumerate(self.allSessions):
                f.write("!!Session\n")
                di = bb.__dict__
                pbp = di.pop('prevSession', None)
                nbp = di.pop('nextSession', None)
                date = di.pop('date', None)

                bb_pfx = bb.name
                clean_di = self.filterAndSaveDict(di, bb_pfx, np_dir)
                f.write(json.dumps(clean_di) + '\n')

                di['prevSession'] = pbp
                di['nextSession'] = nbp
                di['date'] = date

            for rsi, rs in enumerate(self.allRestSessions):
                f.write("!!RestSession\n")
                di = rs.__dict__
                date = di.pop('date', None)
                btwp = di.pop('btwpSession', None)

                rs_pfx = rs.name
                clean_di = self.filterAndSaveDict(di, rs_pfx, np_dir)
                f.write(json.dumps(clean_di) + '\n')

                di['date'] = date
                di['btwpSession'] = btwp

            self.filename = filename
            return 0
        print("Couldn't open file %s" % filename)
        return -1

    def filterAndSaveDict(self, di, file_pfx, np_dir):
        clean_di = dict()
        for k, v in di.items():
            if is_jsonable(v):
                clean_di[k] = v
                saved = True
            elif 'numpy' in str(type(v)):
                kk = NP_KEY_PFX + k
                vv = file_pfx + "__" + k + ".npy"
                clean_di[kk] = vv
                np.save(os.path.join(np_dir, vv), v)
                saved = True
            elif 'list' in str(type(v)):  # check for list of numpy objects too
                try:
                    ar = np.array(v)
                    kk = NP_LIST_KEY_PFX + k
                    vv = file_pfx + "__" + k + ".npy"
                    clean_di[kk] = vv
                    np.save(os.path.join(np_dir, vv), ar, allow_pickle=True)
                    saved = True
                except Exception:
                    saved = False
            else:
                saved = False

            if not saved:
                print("WARNING: member variable {0} of type {1} is not json serializable. It is not being saved!".format(
                    k, type(v)))

        return clean_di

    def processAndLoadDict(self, load_dict, np_dir):
        processed_dict = dict()
        for k, v in load_dict.items():
            if NP_KEY_PFX in k:
                kk = k.split(NP_KEY_PFX)[1]
                vv = np.load(os.path.join(np_dir, v))
                processed_dict[kk] = vv
            elif NP_LIST_KEY_PFX in k:
                kk = k.split(NP_LIST_KEY_PFX)[1]
                vv = np.load(os.path.join(np_dir, v), allow_pickle=True)
                processed_dict[kk] = list(vv)
            else:
                processed_dict[k] = v

        return processed_dict

    # can pass in optional filter function, otherwise returns all blocks
    def getSessions(self, predicate=lambda b: True):
        return list(filter(predicate, self.allSessions))

    def getRestSessions(self, predicate=lambda b: True):
        return list(filter(predicate, self.allRestSessions))
