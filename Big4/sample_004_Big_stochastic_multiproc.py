# doc/drawing/sample_004_stochastic.svg

from petnetsim import *
import time
import multiprocessing

#import pyjion; pyjion.enable()

petri_net_init = PetriNet([Place('Z', init_tokens=100000), 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'],
                         [TransitionStochastic('T1', 3), TransitionStochastic('T2', 49),
                          TransitionStochastic('T3', 7), TransitionStochastic('T4', 5),
                          TransitionStochastic('T5', 6), TransitionStochastic('T6', 20),
                          TransitionStochastic('T7', 8), TransitionStochastic('T8', 2)],
                         [('Z', 'T1'), ('Z', 'T2'), ('Z', 'T3'), ('Z', 'T4'),('Z', 'T5'),
                          ('T1', 'A'), ('T1', 'B'), ('T1', 'C'), ('T1', 'D'), ('T1', 'E'),
                          ('T1', 'F'), ('T1', 'G'), ('T1', 'H'), ('T1', 'I'), ('T1', 'J'),

                          ('T2', 'A'), ('T2', 'B'), ('T2', 'C'), ('T2', 'D'), ('T2', 'E'),


                          ('T3', 'A'), ('T3', 'B'), ('T3', 'C'), ('T3', 'D'), ('T3', 'E'),
                          ('T3', 'F'), ('T3', 'G'), ('T3', 'H'), ('T3', 'I'), ('T3', 'J'),


                          ('T4', 'F'), ('T4', 'G'), ('T4', 'H'), ('T4', 'I'), ('T4', 'J'),

                          ('T5', 'A'), ('T5', 'B'), ('T5', 'C'), ('T5', 'D'), ('T5', 'E'),
                          ('T5', 'F'), ('T5', 'G'), ('T5', 'H'), ('T5', 'I'), ('T5', 'J'),

                          ('T5', 'A'), ('T5', 'B'), ('T5', 'C'), ('T5', 'D'), ('T5', 'E'),
                          ('T5', 'F'), ('T5', 'G'), ('T5', 'H'), ('T5', 'I'), ('T5', 'J'),

                          ('A', 'T6'), ('B', 'T6'), ('C', 'T6'), ('D', 'T6'), ('E', 'T6'),
                          ('A', 'T7'), ('B', 'T7'), ('C', 'T7'), ('D', 'T7'), ('E', 'T7'),

                          ('F', 'T8'), ('G', 'T8'), ('H', 'T8'), ('I', 'T8'), ('J', 'T8'),

                          ('T6', 'K'), ('T6', 'M'), ('T6', 'O'),
                          ('T7', 'K'), ('T7', 'M'), ('T7', 'O'),
                          ('T8', 'K'), ('T8', 'L'), ('T8', 'M'), ('T8', 'N'), ('T8', 'O')

                          ])
def run():
    petri_net = petri_net_init

    print('conflict groups:', petri_net.conflict_groups_str)

    print('------------------------------------')
    print(' run')


    petri_net.reset()

    max_steps = 10000000

    print('--------------- step', petri_net.step_num)
    petri_net.print_places()

    while not petri_net.ended and petri_net.step_num < max_steps:
        petri_net.step()
        #print('--------------- step', petri_net.step_num)
        #petri_net.print_places()

    if petri_net.ended:
        print('  breaking condition')
    else:
        print('  max steps reached')

    print('transitions stats')
    for t in petri_net.transitions:
        print(t.name, t.fired_times, sep=': ')
        petri_net.print_places()

cyc_num = 5
"""""
i=1
Time_001 = []
while i < cyc_num:
    start_time = time.perf_counter()

    run()
    Time_001.append(time.perf_counter() - start_time)
    i = i+1
"""""

start_time = time.perf_counter()
processes = []
for _ in range(20):
    p = multiprocessing.Process(target=run)
    p.start()
    processes.append(p)

for process in processes:
    process.join()

print(time.perf_counter() - start_time)

"""""
np.savetxt("Big4_pyston__print.csv",
           Time_001,
           delimiter =", ",
           fmt ='% s')
"""""