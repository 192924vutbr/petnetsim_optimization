
from samples_noprint import sample_001_basic_edit, sample_002_conflict_groups,sample_003_inhibitors_edit, sample_004_stochastic, sample_005_priority, sample_006_timed, sample_007_deadlock, sample_008_deadlock_priority, sample_009_deadlock_P1watchdog, sample_010_capacity_limit_shared_output, sample_011_json, sample_012_netclone
import time
import numpy as np
import csv
import pyjion; pyjion.enable()

cyc_num = 20
i=1
Time_001 = []
while i < cyc_num:
    start_time = time.time()

    sample_001_basic_edit.run()
    Time_001.append(time.time() - start_time)
    i = i+1
i=1
Time_002 = []

while i < cyc_num:
    start_time = time.time()

    sample_002_conflict_groups.run()
    Time_002.append(time.time() - start_time)
    i = i+1
i=1
Time_003 = []
while i < cyc_num:
    start_time = time.time()

    sample_003_inhibitors_edit.run()
    Time_003.append(time.time() - start_time)
    i = i + 1
i=1
Time_004 = []
while i < cyc_num:
    start_time = time.time()

    sample_004_stochastic.run()
    Time_004.append(time.time() - start_time)
    i = i + 1
i=1
Time_005 = []
while i < cyc_num:
    start_time = time.time()

    sample_005_priority.run()
    Time_005.append(time.time() - start_time)
    i = i + 1
i=1
Time_006 = []
while i < cyc_num:
    start_time = time.time()

    sample_006_timed.run()
    Time_006.append(time.time() - start_time)
    i = i + 1
i=1
Time_007 = []
while i < cyc_num:
    start_time = time.time()

    sample_007_deadlock.run()
    Time_007.append(time.time() - start_time)
    i = i + 1
i=1
Time_008 = []
while i < cyc_num:
    start_time = time.time()

    sample_008_deadlock_priority.run()
    Time_008.append(time.time() - start_time)
    i = i + 1
i=1
Time_009 = []
while i < cyc_num:
    start_time = time.time()

    sample_009_deadlock_P1watchdog.run()
    Time_009.append(time.time() - start_time)
    i = i + 1
i=1
Time_010 = []
while i < cyc_num:
    start_time = time.time()

    sample_010_capacity_limit_shared_output.run()
    Time_010.append(time.time() - start_time)
    i = i + 1

i=1
Time_011 = []
while i < cyc_num:
    start_time = time.time()

    sample_011_json.run()
    Time_011.append(time.time() - start_time)
    i = i + 1

i=1
Time_012 = []
while i < cyc_num:
    start_time = time.time()

    sample_012_netclone.run()
    Time_012.append(time.time() - start_time)
    i = i + 1
fields = ["Time_001", "Time_002", "Time_003", "Time_004", "Time_005", "Time_006", "Time_007", "Time_008", "Time_009", "Time_010", "Time_011," "Time_012"]
Python_TIME = []
Python_TIME.append(Time_001)
Python_TIME.append(Time_002)
Python_TIME.append(Time_003)
Python_TIME.append(Time_004)
Python_TIME.append(Time_005)
Python_TIME.append(Time_006)
Python_TIME.append(Time_007)
Python_TIME.append(Time_008)
Python_TIME.append(Time_009)
Python_TIME.append(Time_010)
Python_TIME.append(Time_011)
Python_TIME.append(Time_012)

np.savetxt("Pyjion_noprint.csv",
           Python_TIME,
           delimiter =", ",
           fmt ='% s')