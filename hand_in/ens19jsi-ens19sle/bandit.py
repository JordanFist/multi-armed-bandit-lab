# epsilon-greedy example implementation of a multi-armed bandit
import random
from math import sqrt

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import simulator
import reference_bandit

# generic epsilon-greedy bandit
class Bandit:
    def __init__(self, arms, epsilon=0.6009451241232561, x=0.8972644216263362):
        self.arms = arms
        self.epsilon = epsilon
        self.x = x
        self.nbRound = 0
        self.treshold = 0.5
        self.frequencies = [0] * len(arms)
        self.bestFrequencies = [0] * len(arms)
        self.sums = [0] * len(arms)
        self.expected_values = [0] * len(arms)

    def run(self):
        self.nbRound += 1
        if self.nbRound > 50:
            self.removeArm()
        if min(self.bestFrequencies) == 0:
            return self.arms[self.bestFrequencies.index(min(self.bestFrequencies))]
        self.epsilon *= self.x
        if random.random() < self.epsilon:
            return self.arms[random.randint(0, len(self.arms) - 1)]
        return self.arms[self.expected_values.index(max(self.expected_values))]

    def give_feedback(self, arm, reward):
        arm_index = self.arms.index(arm)
        sum = self.sums[arm_index] + reward
        self.sums[arm_index] = sum
        frequency = self.bestFrequencies[arm_index] + 1
        self.bestFrequencies[arm_index] = frequency
        self.frequencies[arm_index] += 1
        expected_value = sum / frequency
        self.expected_values[arm_index] = expected_value

    def getStandardDeviations(self):
        res = []
        mean = sum(self.expected_values) / len(self.arms)
        for i in range(len(self.arms)):
            sd = 0
            for j in range(len(self.arms)):
                if j != i:
                    sd += self.expected_values[j] - mean
            sd /= len(self.arms)
            res.append(sd)
        return res

    def removeArm(self):
        sd = self.getStandardDeviations()
        for i in range(len(sd)):
            if sd[i] > self.treshold:
                del self.arms[i]
                del self.bestFrequencies[i]
                del self.sums[i]
                del self.expected_values[i]
                return


# configuration
arms = [
    'Configuration a',
    'Configuration b',
    'Configuration c',
    'Configuration d',
    'Configuration e',
    'Configuration f'
]

bandit = Bandit(arms)
ref_bandit = reference_bandit.ReferenceBandit(arms.copy())
