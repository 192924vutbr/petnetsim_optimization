# doc/drawing/sample_006_timed.svg

import petnetsim_opt as pns


def run():

    petri_net = pns.PetriNet([pns.Place('X', init_tokens=4), pns.Place('Y', init_tokens=12), pns.Place('Z', init_tokens=2),
                          'A', 'B', 'C', 'D'],
                         [pns.TransitionTimed('T1', 10), pns.TransitionTimed('T2', 5),
                          pns.TransitionTimed('T3', 1), 'T4'],
                         [('X', 'T1'), ('X', 'T2'), ('Y', 'T3'), ('Y', 'T4'), ('Z', 'T4'),
                          ('T1', 'A'), ('T2', 'B'), ('T3', 'C'), ('T4', 'D')])

    print('------------------------------------')
    print(' run')

    petri_net.reset()

    max_steps = 50

    print('--------------- step', petri_net.step_num, '   t:', petri_net.time)
    petri_net.print_places()

    print('step', 't', 'fired', sep='\t')

    while not petri_net.ended and petri_net.step_num < max_steps:
        petri_net.step()
        print('--------------- step', petri_net.step_num, '   t:', petri_net.time)
        if len(petri_net.fired):
            print(' fired: ', end='')
            print(*(t.name for t in petri_net.fired), sep=', ')

        petri_net.print_places()

        print(petri_net.step_num, petri_net.time, ', '.join(t.name for t in petri_net.fired), sep='\t')


    if petri_net.ended:
        print('  breaking condition')
    else:
        print('  max steps reached')

    print('transitions stats')
    for t in petri_net.transitions:
        print(t.name, t.fired_times, sep=': ')


run()
