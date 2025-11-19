# Jamie Roberson for CS330 FA-25

import random

class Transition:
    id = 1
    def __init__(self, from_id: int, to_id: int, probability: float):
        self.id = Transition.id
        Transition.id += 1
        self.from_id = from_id
        self.to_id = to_id
        self.prob = probability

class StateManager:
    def __init__(self, transitions: list[Transition], states: list[str]):
        self.state = 1
        self.transitions = transitions
        self.states = states

    def update(self):
        val = random.random()
        high = 0.0
        for transition in self.transitions:
            if transition.from_id != self.state:
                continue
        
            high += transition.prob
            if val < high:
                # Transition event!
                self.state = transition.to_id
                return transition.id

        return None

    def reset(self):
        self.state = 1

    def get_string_state(self) -> str:
        return self.states[self.state-1]
    
    def get_probabilities(self) -> list[float]:
        prob_list = []
        for transition in self.transitions:
            prob_list.append(transition.prob)

        return prob_list

def run_iteration(state_manager: StateManager, f, trace=True) -> tuple[list[int], list[int]]:
    state_manager.reset()
    state_counts = len(state_manager.states) * [0]
    transition_counts = len(state_manager.transitions) * [0]

    while True:
        if trace:
            print(f'state= {state_manager.state} {state_manager.get_string_state()}')
            f.write(f'state= {state_manager.state} {state_manager.get_string_state()}\n')

        state_counts[state_manager.state-1] += 1

        if state_manager.state == 7:
            if trace:
                print()
                f.write('\n')
            break

        transition = state_manager.update()
        if transition is None:
            continue

        transition_counts[transition-1] += 1

    return state_counts, transition_counts

def calculate_frequencies(arr: list[int]) -> list[float]:
    frequencies = []
    total_count = sum(arr)
    for item in arr:
        frequencies.append(item/total_count)
    return frequencies

if __name__ == '__main__':
    running_scenario_one = True
    states = ['Follow', 'Pull out', 'Accelerate', 'Pull in ahead', 'Pull in behind', 'Decelerate', 'Done']
    transitions: list[Transition] = []

    if running_scenario_one:
        filename = 'scenario_one_output.txt'
        transitions.append(Transition(1, 2, 0.8))
        transitions.append(Transition(2, 3, 0.4))
        transitions.append(Transition(3, 4, 0.3))
        transitions.append(Transition(2, 5, 0.4))
        transitions.append(Transition(3, 5, 0.3))
        transitions.append(Transition(3, 6, 0.3))
        transitions.append(Transition(5, 1, 0.8))
        transitions.append(Transition(6, 5, 0.8))
        transitions.append(Transition(4, 7, 0.8))
        iterations = 100

    else: # running scenario two
        filename = 'scenario_two_output.txt'
        transitions.append(Transition(1, 2, 0.9))
        transitions.append(Transition(2, 3, 0.6))
        transitions.append(Transition(3, 4, 0.3))
        transitions.append(Transition(2, 5, 0.2))
        transitions.append(Transition(3, 5, 0.2))
        transitions.append(Transition(3, 6, 0.4))
        transitions.append(Transition(5, 1, 0.7))
        transitions.append(Transition(6, 5, 0.9))
        transitions.append(Transition(4, 7, 0.7))
        iterations = 1_000_000

    state_manager = StateManager(transitions, states)

    state_counts = len(states) * [0]
    trans_counts = len(transitions) * [0]

    with open(filename, 'w') as f:
        for i in range(iterations):
            if running_scenario_one:
                print(f'iteration= {i+1}')
                f.write(f'iteration= {i+1}\n')

            s, t = run_iteration(state_manager, f, trace=running_scenario_one)

            for j, val in enumerate(s):
                state_counts[j] += val
            for k, val in enumerate(t):
                trans_counts[k] += val

        print(f'scenario                = 1')
        print(f'trace                   = TRUE')
        print(f'iterations              = {iterations}')
        print(f'transition probabilities= {state_manager.get_probabilities()}')
        print(f'state counts            = {state_counts}')
        print(f'state frequencies       = {calculate_frequencies(state_counts)}')
        print(f'transition counts       = {trans_counts}')
        print(f'transition frequencies  = {calculate_frequencies(trans_counts)}')

        f.write(f'scenario                = 1\n')
        f.write(f'trace                   = {"TRUE" if running_scenario_one else "FALSE"}\n')
        f.write(f'iterations              = {iterations}\n')
        f.write(f'transition probabilities= {state_manager.get_probabilities()}\n')
        f.write(f'state counts            = {state_counts}\n')
        f.write(f'state frequencies       = {calculate_frequencies(state_counts)}\n')
        f.write(f'transition counts       = {trans_counts}\n')
        f.write(f'transition frequencies  = {calculate_frequencies(trans_counts)}\n')