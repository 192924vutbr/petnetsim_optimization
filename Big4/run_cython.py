import sample_004_Big_stochastic_multiproc
import time
import numpy as np

cyc_num = 21
i=1
Time_001 = []
while i < cyc_num:
    start_time = time.perf_counter()
    sample_004_Big_stochastic.run()

    Time_001.append(time.perf_counter() - start_time)
    i = i+1


np.savetxt("Big4_cython__print.csv",
           Time_001,
           delimiter =", ",
           fmt ='% s')