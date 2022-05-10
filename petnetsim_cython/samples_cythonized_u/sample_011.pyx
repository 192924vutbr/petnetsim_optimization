
from petnetsim.json_io import loads, dumps
from pprint import pprint
from itertools import chain
import json
import random as _random
from copy import copy, deepcopy
import re
from enum import IntEnum
import sys
import numpy as np

_default_context_init = {
    'counters': {'P': 1, 'T': 1, 'A': 1, 'I': 1}
}

_default_context = deepcopy(_default_context_init)


def default_context():
    return _default_context


def new_context():
    return deepcopy(_default_context_init)


def reset_default_context():
    # preserve _default_context as same object, but deepcopy all included
    _default_context.clear()
    _default_context.update({k: deepcopy(v) for k, v in _default_context_init.items()})


class Place:
    INF_CAPACITY = 0

    def __init__(self, name=None, init_tokens=0, capacity=INF_CAPACITY, context=default_context()):
        if name is None:
            self.name = 'P_' + str(context['counters']['P'])
            context['counters']['P'] += 1
        else:
            match = re.fullmatch(r'P_(\d+)', name)
            if match is not None:
                context['counters']['P'] = int(match.group(1)) + 1

            self.name = name
        self.capacity = capacity
        self.init_tokens = init_tokens
        self.tokens = init_tokens

    def can_add(self, n_tokens):
        return self.capacity == Place.INF_CAPACITY or self.tokens + n_tokens <= self.capacity

    def can_remove(self, n_tokens):
        return self.tokens - n_tokens >= 0

    def add(self, n_tokens):
        self.tokens += n_tokens

    def remove(self, n_tokens):
        self.tokens -= n_tokens

    def reset(self):
        self.tokens = self.init_tokens

    def clone(self, prefix):
        p = copy(self)
        p.name = prefix + p.name
        return p


class Transition:
    def __init__(self, name, context=default_context()):
        if name is None:
            self.name = 'T_' + str(context['counters']['T'])
            context['counters']['T'] += 1
        else:
            match = re.fullmatch(r'T_(\d+)', name)
            if match is not None:
                context['counters']['T'] = int(match.group(1)) + 1
            self.name = name
        self.inputs = set()  # Arc, Inhibitor
        self.outputs = set()  # Arc, Inhibitor
        self.fired_times = 0
        self.in_arcs = []  # init in reset
        self.inhibitors = []  # init in reset

    def output_possible(self):
        return all(arc.target.can_add(arc.n_tokens) for arc in self.outputs)

    def enabled(self):
        return all(arc.source.can_remove(arc.n_tokens) for arc in self.in_arcs) \
               and self.output_possible() \
               and not any(inhibitor.source.can_remove(inhibitor.n_tokens) for inhibitor in self.inhibitors)

    def fire(self):
        for arc in self.in_arcs:
            arc.source.remove(arc.n_tokens)

        for arc in self.outputs:
            arc.target.add(arc.n_tokens)

        self.fired_times += 1

    def freeze(self):
        # todo no freezing for editing!
        # self.inputs = frozenset(self.inputs)
        # self.outputs = frozenset(self.outputs)
        self.in_arcs = tuple(arc for arc in self.inputs if isinstance(arc, Arc))
        self.inhibitors = tuple(inhibitor for inhibitor in self.inputs if isinstance(inhibitor, Inhibitor))
        # note: inhibitors can't be outputs

    def reset(self):
        self.fired_times = 0

    def clone(self, prefix):
        return Transition(prefix + self.name)


class TransitionPriority(Transition):
    def __init__(self, name, priority, context=default_context()):
        super().__init__(name, context)
        self.priority = priority

    def clone(self, prefix):
        return TransitionPriority(prefix + self.name, self.priority)


def constant_distribution(t_min, t_max):
    return t_min


def uniform_distribution(t_min, t_max):
    return _random.uniform(t_min, t_max)


class TransitionTimed(Transition):
    T_EPSILON = 1e6

    def __init__(self, name, t_min, t_max=1, p_distribution_func=constant_distribution, context=default_context()):
        super().__init__(name, context)
        self.remaining = 0
        self.t_min = t_min
        self.t_max = t_max
        self.p_distribution_func = p_distribution_func
        self.is_waiting = False
        self.time = 0.0

    def enabled(self):
        return super().enabled() and not self.is_waiting

    def choose_time(self):
        self.time = self.p_distribution_func(self.t_min, self.t_max)
        return self.time

    def fire(self):
        for arc in self.in_arcs:
            arc.source.remove(arc.n_tokens)
        self.is_waiting = True
        self.fired_times += 1

    def fire_phase2(self):
        for arc in self.outputs:
            arc.target.add(arc.n_tokens)
        self.is_waiting = False

    def reset(self):
        super().reset()
        self.is_waiting = False

    def dist_time_str(self):
        if self.p_distribution_func is constant_distribution:
            return f"{self.t_min:.3f}s"
        elif self.p_distribution_func is uniform_distribution:
            return f"U({self.t_min:.3f}~{self.t_max:.3f})s"
        return f"(CustomPDist)"

    def clone(self, prefix):
        return TransitionTimed(prefix + self.name, self.t_min, self.t_max, self.p_distribution_func)


class TransitionStochastic(Transition):
    # NOTE: stochastic is almost normal transition
    def __init__(self, name, probability, context=default_context()):
        super().__init__(name, context)
        self.probability = probability

    def clone(self, prefix):
        return TransitionStochastic(prefix + self.name, self.probability)


class Arc:
    def __init__(self, source, target, n_tokens=1, name=None, context=default_context()):
        if name is None:
            self.name = 'Arc_' + str(context['counters']['A'])
            context['counters']['A'] += 1
        else:
            match = re.fullmatch(r'Arc_(\d+)', name)
            if match is not None:
                context['counters']['A'] = int(match.group(1)) + 1
            self.name = name
        self.source = source
        self.target = target
        self.n_tokens = n_tokens

        if not (isinstance(self.source, str) or isinstance(self.target, str)):
            self.connect(None)

    def to_inhibitor(self, context=default_context()):
        return Inhibitor(self.source, self.target, self.n_tokens, self.name, context)

    def connect(self, names_lookup):
        if isinstance(self.source, str):
            self.source = names_lookup[self.source]
        if isinstance(self.target, str):
            self.target = names_lookup[self.target]

        if isinstance(self.source, Transition):
            if not isinstance(self.target, Place):
                raise RuntimeError('arc from Transition must go to a Place')
            self.source.outputs.add(self)
        if isinstance(self.target, Transition):
            if not isinstance(self.source, Place):
                raise RuntimeError('arc to Transition must go from a Place')
            self.target.inputs.add(self)

    @property
    def target_infinite_capacity(self):
        if not isinstance(self.target, Place):
            raise RuntimeError('target_is_infinite can be asked only if target is a Place')
        return self.target.capacity == Place.INF_CAPACITY


class Inhibitor:
    def __init__(self, source, target, n_tokens=1, name=None, context=default_context()):
        if name is None:
            self.name = 'Inhibitor_' + str(context['counters']['I'])
            context['counters']['I'] += 1
        else:
            match = re.fullmatch(r'Inhibitor_(\d+)', name)
            if match is not None:
                context['counters']['I'] = int(match.group(1)) + 1
            self.name = name
        self.source = source
        self.target = target
        self.n_tokens = n_tokens

        if not (isinstance(self.source, str) or isinstance(self.target, str)):
            self.connect(None)

    def to_arc(self, context=default_context()):
        return Arc(self.source, self.target, self.n_tokens, self.name, context)

    def connect(self, names_lookup):
        if isinstance(self.source, str):
            self.source = names_lookup[self.source]
        if isinstance(self.target, str):
            self.target = names_lookup[self.target]

        if not isinstance(self.source, Place):
            raise TypeError('inhibitor source must be a Place')

        if isinstance(self.target, Transition):
            self.target.inputs.add(self)
        else:
            raise RuntimeError('inhibitor target must be a Transition')


class ConflictGroupType(IntEnum):
    Normal = 0
    Priority = 1
    Stochastic = 2
    Timed = 3


class PetriNet:
    def __init__(self, places, transitions, arcs, context=default_context()):
        self._names_lookup = {}

        places = [Place(p, context=context) if isinstance(p, str) else p for p in places]

        for p in places:
            if p.name in self._names_lookup:
                raise RuntimeError('name reused: ' + p.name)
            self._names_lookup[p.name] = p

        transitions = [Transition(t, context=context) if isinstance(t, str) else t
                       for t in transitions]

        for t in transitions:
            if t.name in self._names_lookup:
                raise RuntimeError('name reused: ' + t.name)
            self._names_lookup[t.name] = t

        def get_i(obj, i, default=1):

            try:
                v = obj[i]
            except TypeError:
                v = default
            except IndexError:
                v = default
            return v

        arcs = [Arc(a[0], a[1], get_i(a, 2), get_i(a, 3, None), context=context)
                if isinstance(a, (tuple, list)) else a
                for a in arcs]

        for arc in arcs:
            if arc.name in self._names_lookup:
                raise RuntimeError('name reused: ' + arc.name)
            self._names_lookup[arc.name] = arc
            arc.connect(self._names_lookup)

        for t in transitions:
            t.freeze()

        self.places = tuple(places)
        self.transitions = tuple(transitions)
        self.arcs = tuple(arcs)

        self._make_conflict_groups()

        self.enabled = np.zeros(len(transitions), dtype=np.bool_)
        self.enabled_tmp = np.zeros(len(transitions), dtype=np.bool_)
        self._ended = False
        self.step_num = 0
        self.time = 0.0
        # fired in last step
        self.fired = []
        self.fired_phase2 = []

    def clone(self, prefix: str, places, transitions, arcs, context=default_context()):
        for p in self.places:
            places.append(p.clone(prefix))
        for t in self.transitions:
            transitions.append(t.clone(prefix))
        for a in self.arcs:
            if isinstance(a, Arc):
                arcs.append((prefix + a.source.name, prefix + a.target.name, a.n_tokens, prefix + a.name))
            elif isinstance(a, Inhibitor):
                arcs.append(Inhibitor(prefix + a.source.name, prefix + a.target.name, a.n_tokens, prefix + a.name))
            else:
                raise TypeError(f'cannot handle type: {type(a)}')

    @property
    def ended(self):
        return self._ended

    def reset(self):
        self._ended = False
        self.step_num = 0
        self.time = 0.0
        self.fired.clear()
        self.conflict_groups_waiting.fill(0)
        for t in self.transitions:
            t.reset()
        for p in self.places:
            p.reset()

    def step(self, record_fired=True):
        if record_fired:
            self.fired.clear()
            self.fired_phase2.clear()
        # enabled transitions
        for ti, t in enumerate(self.transitions):
            self.enabled[ti] = t.enabled()

        CGT = ConflictGroupType

        num_fired = 0
        enabled_any = self.enabled.any()
        if enabled_any:
            np.bitwise_and(self.enabled, self.conflict_groups_mask, out=self.enabled_conflict_groups)

            for cgi, ecg in enumerate(self.enabled_conflict_groups):
                if ecg.any():
                    cg_type = self.conflict_groups_types[cgi]
                    t_idxs = np.argwhere(ecg).flatten()  # absolute indices of enabled transitions in group
                    t_fire_idx = None
                    if cg_type == CGT.Normal:
                        t_fire_idx = np.random.choice(t_idxs)
                    elif cg_type == CGT.Priority:
                        priorities = self.conflict_group_data[cgi]
                        ep = priorities[t_idxs]
                        ep_idxs = np.argwhere(ep == ep.max()).flatten()
                        ep_idx = np.random.choice(ep_idxs)
                        t_fire_idx = t_idxs[ep_idx]
                    elif cg_type == CGT.Stochastic:
                        probabilities = self.conflict_group_data[cgi][t_idxs]
                        # "normalize" the sum of probabilities to 1
                        probabilities_norm = probabilities * (1 / np.sum(probabilities))
                        t_fire_idx = np.random.choice(t_idxs, p=probabilities_norm)
                    elif cg_type == CGT.Timed:
                        # conflict_group_data[cgi][0, ti] = isinstance(t, TransitionTimed)
                        # conflict_group_data[cgi][1, ti] = not isinstance(t, TransitionTimed)
                        if self.conflict_groups_waiting[cgi] <= 0:
                            normal_enabled = self.enabled_tmp
                            np.bitwise_and(ecg, self.conflict_group_data[cgi][1], out=normal_enabled)
                            if any(normal_enabled):  # use normal transition
                                normal_t_idxs = np.argwhere(normal_enabled).flatten()
                                t_fire_idx = np.random.choice(normal_t_idxs)
                            else:  # then must be timed
                                timed_enabled = self.enabled_tmp
                                np.bitwise_and(ecg, self.conflict_group_data[cgi][0], out=timed_enabled)
                                timed_t_idxs = np.argwhere(timed_enabled).flatten()
                                timed_t_idx = np.random.choice(timed_t_idxs)
                                t_fire_idx = timed_t_idx
                                timed_t: TransitionTimed = self.transitions[timed_t_idx]
                                self.conflict_groups_waiting[cgi] = timed_t.choose_time()
                                # print(' ', timed_t.name, 'wait =', self.conflict_groups_waiting[cgi])

                    if t_fire_idx is not None:
                        t = self.transitions[t_fire_idx]
                        if t.output_possible():
                            t.fire()
                            num_fired += 1
                            if record_fired:
                                self.fired.append(t)
                        else:
                            print(f'warning: transition "{t.name}" was enabled, but output not possible',
                                  file=sys.stderr)

        num_waiting = np.sum(self.conflict_groups_waiting > 0)

        if num_waiting > 0 and num_fired == 0:
            # nothing fired -> advance time and fire waiting timed transitions
            min_time = np.min(self.conflict_groups_waiting[self.conflict_groups_waiting > 0])
            self.time += min_time

            for cgi in np.argwhere(self.conflict_groups_waiting == min_time).flatten():
                for ti in np.where(self.conflict_group_data[cgi][0])[0]:
                    t: TransitionTimed = self.transitions[ti]
                    if t.is_waiting:
                        if t.output_possible():
                            t.fire_phase2()
                            self.fired_phase2.append(t)
                            break
                        else:
                            msg = f'timed transition "{t.name}" was fired, but output not possible for phase 2'
                            raise RuntimeError(msg)

                self.conflict_groups_waiting[cgi] = 0

            np.subtract(self.conflict_groups_waiting, min_time, out=self.conflict_groups_waiting)
            np.clip(self.conflict_groups_waiting, 0, float('inf'), out=self.conflict_groups_waiting)

        if not enabled_any and num_waiting == 0:
            self._ended = True
        self.step_num += 1

    def print_places(self):
        for p in self.places:
            print(p.name, p.tokens, sep=': ')

    def print_all(self):
        print('places:')
        for p in self.places:
            print('  ', p.name, p.init_tokens)
        print('transitions:')
        for t in self.transitions:
            print('  ', t.name, t.__class__.__name__)
        print('arcs:')
        for a in self.arcs:
            print('  ' if type(a) == Arc else ' I', a.name, a.target.name, '--' + str(a.n_tokens) + '->', a.source.name)

    def validate(self):
        # TODO : validation of whole net
        print('TODO: PetriNet.validate')
        pass

    @property
    def conflict_groups_str(self):
        return ', '.join('{' + ', '.join(sorted(t.name for t in s)) + '}' for s in self.conflict_groups_sets)

    def _make_conflict_groups(self):
        conflict_groups_sets = [{self.transitions[0]}] if len(self.transitions) else []
        for t in self.transitions[1:]:
            add_to_cg = False
            # print('t: ', t.name)
            for cg in conflict_groups_sets:
                for cg_t in cg:
                    # ignore inhibitors!
                    t_in = set(arc.source for arc in t.inputs if isinstance(arc, Arc))
                    # t_out = set(
                    #    arc.target for arc in t.outputs if isinstance(arc, Arc) and not arc.target_infinite_capacity)
                    cg_t_in = set(arc.source for arc in cg_t.inputs if isinstance(arc, Arc))
                    # cg_t_out = set(
                    #    arc.target for arc in cg_t.outputs if isinstance(arc, Arc) and not arc.target_infinite_capacity)

                    # [NOTE] outputs collision ignored

                    add_to_cg = add_to_cg or not t_in.isdisjoint(cg_t_in)
                    # add_to_cg = add_to_cg or not t_out.isdisjoint(cg_t_out)
                    if add_to_cg:
                        break
                if add_to_cg:
                    cg.add(t)
                    break

            if not add_to_cg:
                conflict_groups_sets.append({t})

        conflict_groups_types = [None for _ in conflict_groups_sets]

        def t_cg_type(transition):
            if isinstance(transition, TransitionPriority):
                return ConflictGroupType.Priority
            elif isinstance(transition, TransitionStochastic):
                return ConflictGroupType.Stochastic
            elif isinstance(transition, TransitionTimed):
                return ConflictGroupType.Timed
            return ConflictGroupType.Normal

        CGT = ConflictGroupType
        conflict_group_data = [None for _ in conflict_groups_sets]
        for cg_i, cg in enumerate(conflict_groups_sets):
            # cg type preferred by the transition
            t_types = [t_cg_type(t) for t in cg]

            if all(tt == CGT.Normal for tt in t_types):
                cg_type = CGT.Normal
            elif all(tt == CGT.Normal or tt == CGT.Priority for tt in t_types):
                # priority can be mixed with Normal
                cg_type = CGT.Priority
                conflict_group_data[cg_i] = np.zeros(len(self.transitions), dtype=np.uint)
            elif all(tt == CGT.Normal or tt == CGT.Timed for tt in t_types):
                # Timed can be mixed with Normal
                cg_type = CGT.Timed
                conflict_group_data[cg_i] = np.zeros((2, len(self.transitions)), dtype=np.bool_)
            elif all(tt == CGT.Stochastic for tt in t_types):
                group_members_names = ', '.join([t.name for t in cg])
                # stochastic are on their own
                cg_type = CGT.Stochastic
                one_t_in_cg = next(iter(cg))
                ot_sources = set(i.source for i in one_t_in_cg.inputs)
                if not all(set(i.source for i in t.inputs) == ot_sources for t in cg):
                    raise RuntimeError(
                        'all members of stochastic group must share the same inputs: ' + group_members_names)

                # TODO: maybe optional feature - all transitions in stochastic group might be required to take same amount of tokens?
                # if not all(t.inputs.n_tokens == one_t_in_cg.inputs.n_tokens for t in cg):
                #    RuntimeError('all members of stochastic group must take same number of tokens:'+group_members_names)

                conflict_group_data[cg_i] = np.zeros(len(self.transitions))
            else:
                raise RuntimeError('Unsupported combination of transitions: ' + ', '.join([str(tt) for tt in t_types]) + \
                                   '\n' + '; '.join(c.__class__.__name__ + ': ' + c.name for c in cg))

            conflict_groups_types[cg_i] = cg_type

        self.conflict_groups_waiting = np.zeros(len(conflict_groups_sets))
        self.conflict_groups_sets = tuple(tuple(cg) for cg in conflict_groups_sets)
        self.conflict_groups_types = tuple(conflict_groups_types)
        self.conflict_groups_mask = np.zeros((len(conflict_groups_sets), len(self.transitions)), dtype=np.bool_)
        self.enabled_conflict_groups = np.zeros((len(conflict_groups_sets), len(self.transitions)), dtype=np.bool_)
        for cgi, (cg, cgt) in enumerate(zip(conflict_groups_sets, conflict_groups_types)):
            for ti, t in enumerate(self.transitions):
                t_in_cg = t in cg
                self.conflict_groups_mask[cgi, ti] = t_in_cg

                if t_in_cg:
                    if cgt == CGT.Priority:
                        conflict_group_data[cgi][ti] = t.priority if hasattr(t, 'priority') else 0
                    elif cgt == CGT.Timed:
                        conflict_group_data[cgi][0, ti] = isinstance(t, TransitionTimed)
                        conflict_group_data[cgi][1, ti] = not isinstance(t, TransitionTimed)
                    elif cgt == CGT.Stochastic:
                        conflict_group_data[cgi][ti] = t.probability

        self.conflict_group_data = tuple(conflict_group_data)


def run():
    ctx = new_context()

    unnamed_place_1 = Place(init_tokens=5, context=ctx)
    unnamed_place_2 = Place(init_tokens=10, context=ctx)
    unnamed_place_3 = Place(init_tokens=5, context=ctx)
    unnamed_place_4 = Place(init_tokens=3, context=ctx)
    unnamed_place_5 = Place(context=ctx)
    unnamed_place_6 = Place(context=ctx)

    unnamed_transition_1 = TransitionPriority(name=None, priority=1, context=ctx)
    unnamed_transition_2 = TransitionPriority(name=None, priority=2, context=ctx)
    unnamed_transition_3 = TransitionStochastic(name=None, probability=0.7, context=ctx)
    unnamed_transition_4 = TransitionTimed(name=None, t_min=2, context=ctx)
    unnamed_transition_5 = TransitionTimed(name=None, t_min=1, t_max=3, p_distribution_func=uniform_distribution,
                                           context=ctx)
    unnamed_transition_6 = Transition(name=None, context=ctx)
    unnamed_transition_7 = Transition(name=None, context=ctx)

    petri_net = \
        PetriNet(
            (Place('A', 8, context=ctx), Place('B', capacity=6, context=ctx),
             Place('C', 10, context=ctx), Place('D', capacity=5, context=ctx), 'E',
             unnamed_place_1, 'F', 'G', unnamed_place_2, unnamed_place_5, unnamed_place_6,
             unnamed_place_3, unnamed_place_4, 'H', 'I'),
            ('T1', unnamed_transition_1, unnamed_transition_2,
             TransitionStochastic('TX', 0.3, context=ctx),
             unnamed_transition_3, unnamed_transition_4,
             unnamed_transition_5, unnamed_transition_6,
             unnamed_transition_7),
            (('A', 'T1'), ('T1', 'B', 3), ('C', unnamed_transition_1, 2), ('C', unnamed_transition_2),
             (unnamed_transition_1, 'D'), (unnamed_transition_2, 'E'),
             (unnamed_place_1, 'TX'), (unnamed_place_1, unnamed_transition_3),
             ('TX', 'F'), (unnamed_transition_3, 'F'), (unnamed_transition_3, 'G'),
             (unnamed_place_2, unnamed_transition_4), (unnamed_place_2, unnamed_transition_5),
             (unnamed_transition_4, unnamed_place_5), (unnamed_transition_5, unnamed_place_6),
             (unnamed_place_3, unnamed_transition_6, 3), (unnamed_transition_6, 'H'),
             Inhibitor(unnamed_place_3, unnamed_transition_7, context=ctx),
             (unnamed_place_4, unnamed_transition_7), (unnamed_transition_7, 'I')),
            context=ctx)

    petri_net.reset()
    petri_net.step()
    petri_net.step()
    petri_net.step()

    petri_net_dump = dumps(petri_net.places, petri_net.transitions, petri_net.arcs)
    print(petri_net_dump)
    print('-----------------------------------------------')
    with open('sample_011_dump.json', 'w') as fout:
        fout.write(petri_net_dump)

    data = json.loads(petri_net_dump)
    data['places'] = {int(k): v for k, v in data['places'].items()}
    data['transitions'] = {int(k): v for k, v in data['transitions'].items()}
    data['arcs'] = {int(k): v for k, v in data['arcs'].items()}
    print('petri_net_dump (edited):')
    pprint(data, indent=4, compact=True, width=200)

    ctx2 = new_context()

    p2_places, p2_transitions, p2_arcs, _ = loads(petri_net_dump, ctx2)

    petri_net_from_dump = PetriNet(p2_places, p2_transitions, p2_arcs)

    print('petri_net_from_dump')
    petri_net_from_dump.print_all()

    petri_net_from_dump.reset()
    petri_net_from_dump.step()
    petri_net_from_dump.step()
    petri_net_from_dump.step()

    def compare_petri_nets(pn1, pn2):
        nets_difference = []

        def make_lookup(pn: PetriNet):
            return {x.name: x for x in chain(pn.places, pn.transitions, pn.arcs)}

        def items_differ(x1, x2):
            if type(x1) != type(x2):
                return 'type_mismatch', x1.name, str(type(x1), '!=', str(type(x2)))
            if type(x1) == Place:
                if x1.capacity != x2.capacity or x1.init_tokens != x2.init_tokens:
                    return (x1.name, 'place parameters mismatch',
                            (x1.capacity, x1.init_tokens),
                            '!=',
                            (x2.capacity, x2.init_tokens))
            elif isinstance(x1, Transition):  # and subclasses
                # inputs/outputs
                x1_sources = {arc.source.name for arc in x1.inputs}
                x2_sources = {arc.source.name for arc in x2.inputs}
                mismatched_sources = x1_sources.symmetric_difference(x2_sources)

                x1_targets = {arc.target.name for arc in x1.outputs}
                x2_targets = {arc.target.name for arc in x2.outputs}
                mismatched_targets = x1_targets.symmetric_difference(x2_targets)

                if len(mismatched_sources) or len(mismatched_targets):
                    return (x1.name,
                            'transition has mismatched inputs/outputs',
                            ('bad inputs:', mismatched_sources),
                            ('bad outputs:', mismatched_targets))
            elif isinstance(x1, (Arc, Inhibitor)):
                if x1.source.name != x2.source.name or x1.target.name != x2.target.name:
                    return (x1.name,
                            'arc/inhibitor source/target mismatched',
                            (x1.source.name, x2.source.name),
                            (x1.target.name, x2.target.name))

            return False

        lookup_1 = make_lookup(pn1)
        lookup_2 = make_lookup(pn2)

        mismatched_names = set(lookup_1.keys()).symmetric_difference(lookup_2.keys())
        if len(mismatched_names):
            nets_difference.append(('mismatched names', mismatched_names))

        for name, x1 in lookup_1.items():
            x2 = lookup_2[name]
            di = items_differ(x1, x2)
            if di:
                nets_difference.append(di)

        return nets_difference

    diff = compare_petri_nets(petri_net, petri_net_from_dump)
    print('differences in nets:')
    pprint(diff)
