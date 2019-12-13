import random

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import simulator
import reference_bandit

# generic epsilon-greedy bandit
class Bandit:
    # def __init__(self, arms, epsilon=0.6009451241232561, x=0.8972644216263362):
    def __init__(self, arms, epsilon=0.21496921512010675, x=0.9223492240149492):
        self.arms = arms
        self.ignoredArmIndices = set()
        self.epsilon = epsilon
        self.x = x
        self.nbRound = 0
        self.treshold = 0.5
        # frequency array allowing us to save the frequencies for all the arms
        self.frequencies = [0] * len(arms)
        self.bestFrequencies = [0] * len(arms)
        self.sums = [0] * len(arms)
        self.nonCumulativeSums = [0] * len(arms)
        self.expected_values = [0] * len(arms)

    def chooseRandomArm(self):
        armIndices = set(range(len(self.arms)))
        armIndices -= self.ignoredArmIndices
        return random.choice(list(armIndices))

    def chooseBestArm(self):
        armIndices = set(range(len(self.arms)))
        armIndices -= self.ignoredArmIndices
        notIgnoredExpectedValues = [self.expected_values[i] for i in armIndices]
        return self.expected_values.index(max(notIgnoredExpectedValues))

    def reset(self):
        self.ignoredArmIndices = set()
        self.nbRound = 0
        self.bestFrequencies = [0] * len(arms)
        self.nonCumulativeSums = [0] * len(arms)
        self.expected_values = [0] * len(arms)

    def run(self):
        """
        Epsilon decay algorithm with arm removing
        :return: the best arm to pull
        """
        if self.nbRound % 1000 == 0:
            self.reset()
        self.nbRound += 1
        if self.nbRound > 50:
            self.removeArm()
        if min(self.bestFrequencies) == 0:
            return self.arms[self.bestFrequencies.index(min(self.bestFrequencies))]
        self.epsilon *= self.x
        if random.random() < self.epsilon:
            return self.arms[self.chooseRandomArm()]
        return self.arms[self.chooseBestArm()]

    def give_feedback(self, arm, reward):
        """
        :param arm:
        :param reward:
        :return:
        """
        arm_index = self.arms.index(arm)
        sum = self.nonCumulativeSums[arm_index] + reward
        self.nonCumulativeSums[arm_index] = sum
        self.sums[arm_index] += reward
        frequency = self.bestFrequencies[arm_index] + 1
        self.bestFrequencies[arm_index] = frequency
        self.frequencies[arm_index] += 1
        expected_value = sum / frequency
        self.expected_values[arm_index] = expected_value

    def getStandardDeviations(self):
        """
        Here the standard deviation is a custom value taking into consideration the relative distance with the mean
        Computes all the standard deviations of the expected values taking into account all the arms except one
        :return: array of standard deviations
        """
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
        """
        Removes potentially the worst arm depending on the standard deviation value and treshold
        :return:
        """
        sd = self.getStandardDeviations()
        for i in range(len(sd)):
            if sd[i] > self.treshold and i not in self.ignoredArmIndices:
                self.ignoredArmIndices.add(i)
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
ref_bandit = reference_bandit.ReferenceBandit(arms)
