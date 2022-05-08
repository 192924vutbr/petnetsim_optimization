import time
from numba import jit

hasattr(sys, "pyston_version_info")
start_time = time.time()

@jit(nopython=True)
def function():
    total = 0
    for i in range(1, 50000):
        for j in range(1, 50000):
            total += i + j
    return total


total = function()
print(f"The result is {total}")

end_time = time.time()
print(f"It took {end_time-start_time:.2f} seconds to compute")