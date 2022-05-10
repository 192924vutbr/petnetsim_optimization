import sample_001
import sample_002
import sample_003
import sample_004
import sample_005
import sample_006
import sample_007
import sample_008
import sample_009
import sample_010
import sample_011
#import numpy as np

import time

cyc_num = 21
i=1
Time_001 = []
while i < cyc_num:
    start_time = time.time()
    sample_001.run()

    Time_001.append(time.time() - start_time)
    i = i+1

i=1
Time_002 = []
while i < cyc_num:
    start_time = time.time()
    sample_002.run()

    Time_002.append(time.time() - start_time)
    i = i+1

Time_003 = []
while i < cyc_num:
    start_time = time.time()
    sample_003.run()

    Time_003.append(time.time() - start_time)
    i = i+1

Time_004 = []
while i < cyc_num:
    start_time = time.time()
    sample_004.run()

    Time_004.append(time.time() - start_time)
    i = i+1

Time_005 = []
while i < cyc_num:
    start_time = time.time()
    sample_005.run()

    Time_005.append(time.time() - start_time)
    i = i+1

Time_006 = []
while i < cyc_num:
    start_time = time.time()
    sample_006.run()

    Time_006.append(time.time() - start_time)
    i = i+1

Time_007 = []
while i < cyc_num:
    start_time = time.time()
    sample_007.run()

    Time_007.append(time.time() - start_time)
    i = i+1

Time_008 = []
while i < cyc_num:
    start_time = time.time()
    sample_008.run()

    Time_008.append(time.time() - start_time)
    i = i+1

Time_009 = []
while i < cyc_num:
    start_time = time.time()
    sample_009.run()

    Time_009.append(time.time() - start_time)
    i = i+1

Time_010 = []
while i < cyc_num:
    start_time = time.time()
    sample_010.run()

    Time_010.append(time.time() - start_time)
    i = i+1

Time_011 = []
while i < cyc_num:
    start_time = time.time()
    sample_011.run()

    Time_011.append(time.time() - start_time)
    i = i+1


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

print(Python_TIME)
"""""
np.savetxt("Cython_print.csv",
           Python_TIME,
           delimiter =", ",
           fmt ='% s')

"""""