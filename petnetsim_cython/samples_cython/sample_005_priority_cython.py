# doc/drawing/sample_005_priority.svg

import petnetsim_opt as pns


def run():

    arc_defs = [('Z', 'T'+str(i)) for i in range(1, 8)] + \
               [('T' + str(i), chr(j+ord('A'))) for i, j in zip(range(1, 8), range(0, 7))]

    petri_net = pns.PetriNet([pns.Place('Z', init_tokens=25),
                          'A',
                          'B',
                          'C',
                          pns.Place('D', capacity=3),
                          pns.Place('E', capacity=3),
                          pns.Place('F', capacity=3),
                          pns.Place('G', capacity=3)],
                         ['T1',
                          pns.TransitionPriority('T2', 0),
                          pns.TransitionPriority('T3', 0),
                          pns.TransitionPriority('T4', 1),
                          pns.TransitionPriority('T5', 1),
                          pns.TransitionPriority('T6', 2),
                          pns.TransitionPriority('T7', 2)],
                         arc_defs)

    print('------------------------------------')
    print(' run')

    petri_net.reset()

    max_steps = 100000

    print('--------------- step', petri_net.step_num)
    petri_net.print_places()

    while not petri_net.ended and petri_net.step_num < max_steps:
        petri_net.step()
        print('--------------- step', petri_net.step_num)
        petri_net.print_places()

    if petri_net.ended:
        print('  breaking condition')
    else:
        print('  max steps reached')

    print('transitions stats')
    for t in petri_net.transitions:
        print(t.name, t.fired_times, sep=': ')


run()
