# doc/drawing/sample_001_basic.svg

from petnetsim_pyjion import *
import pyjion; pyjion.enable()
def run():
    empty_net = PetriNet([], [], [])
    while not empty_net.ended and empty_net.step_num < 100:
        empty_net.step()

    petri_net = PetriNet([Place('A', init_tokens=5000),
                          Place('B', init_tokens=4000),
                          Place('C', init_tokens=4000),
                          Place('D', init_tokens=4000),
                          Place('E', init_tokens=4000),
                          Place('F', init_tokens=4000),
                          Place('G', init_tokens=4000),
                          Place('H', init_tokens=4000),
                          Place('I', init_tokens=4000),
                          Place('J', init_tokens=4000),
                          Place('A2', init_tokens=5000),
                          Place('B2', init_tokens=4000),
                          Place('C2', init_tokens=4000),
                          Place('D2', init_tokens=4000),
                          Place('E2', init_tokens=4000),
                          Place('F2', init_tokens=4000),
                          Place('G2', init_tokens=4000),
                          Place('H2', init_tokens=4000),
                          Place('I2', init_tokens=4000),
                          Place('J2', init_tokens=4000)
                          ],
                         [Transition('T'),
                          Transition('T2'),
                          Transition('T3'),
                          Transition('T_2'),
                          Transition('T2_2'),
                          Transition('T3_2')
                          ],
                         [Arc('A', 'T', 2),
                          Arc('B', 'T', 1),
                          Arc('T', 'C', 4),
                          Arc('T', 'D', 1),
                          Arc('E', 'T2', 1),
                          Arc('G', 'T2', 1),
                          Arc('T2', 'F', 3),
                          Arc('D', 'T3', 1),
                          Arc('F', 'T3', 1),
                          Arc('T3', 'H', 2),
                          Arc('T3', 'I', 1),
                          Arc('T3', 'J', 1),
                          Arc('A2', 'T_2', 2),
                          Arc('B2', 'T_2', 1),
                          Arc('T_2', 'C2', 4),
                          Arc('T_2', 'D2', 1),
                          Arc('E2', 'T2_2', 1),
                          Arc('G2', 'T2_2', 1),
                          Arc('T2_2', 'F2', 3),
                          Arc('D2', 'T3_2', 1),
                          Arc('F2', 'T3_2', 1),
                          Arc('T3_2', 'H2', 2),
                          Arc('T3_2', 'I2', 1),
                          Arc('T3_2', 'J2', 1)])

    print('------------------------------------')
    print(' run')

    petri_net.reset()

    max_steps = 1000

    #print('--------------- step', petri_net.step_num)
    #petri_net.print_places()

    while not petri_net.ended and petri_net.step_num < max_steps:
        petri_net.step()
        #print('--------------- step', petri_net.step_num)
        #petri_net.print_places()

    if petri_net.ended:
        print('  breaking condition')
    else:
        print('  max steps reached')

