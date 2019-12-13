from bandit import Bandit
from random import random, shuffle, gauss
from math import exp


def generate_reward(arm_index, expected_rewards_approx):
    return gauss(expected_rewards_approx[arm_index], 0.5) + random() / 2

def simulate(bandit):
    """
    :param bandit:
    :return: the regret related to the bandit
    """
    expected_rewards_approx = [
        1 + (random() / 2) for _ in range(4)
    ]

    expected_rewards_approx.append(-5)
    expected_rewards_approx.append(-10)
    shuffle(expected_rewards_approx)

    indexBestArm = expected_rewards_approx.index(max(expected_rewards_approx))
    bestReward = 0

    for _ in range(1000):
        arm = bandit.run()
        reward = generate_reward(bandit.arms.index(arm), expected_rewards_approx)
        bestReward += expected_rewards_approx[indexBestArm] + 0.5
        bandit.give_feedback(arm, reward)
    return bestReward - sum(bandit.sums)

def simultedAnnealing():
    """
    Compares different values of x and epsilon simulating multiple bandits
    :return: the best value for epsilon and x
    """

    temperature = 10000
    decreasing = 0.9991
    maxIteration = 10000
    delta = 0.01

    x = 0.9
    epsilon = 0.6

    minRegret = minX = minEpsilon = float("inf")

    minExp = float("inf")

    arms = [
        'Configuration a',
        'Configuration b',
        'Configuration c',
        'Configuration d',
        'Configuration e',
        'Configuration f'
    ]

    bandit = Bandit(arms, epsilon, x)
    regret = simulate(bandit)
    for _ in range(maxIteration):
        arms = [
            'Configuration a',
            'Configuration b',
            'Configuration c',
            'Configuration d',
            'Configuration e',
            'Configuration f'
        ]
        newEpsilon = max(min(epsilon + epsilon * delta * (random() * 2 - 1), 1), 0)
        newX = max(min(x + x * delta * (random() * 2 - 1), 1), 0)
        bandit = Bandit(arms, newEpsilon, newX)
        newRegret = simulate(bandit)
        deltaR = newRegret - regret
        if random() < exp(- deltaR / temperature):
            regret = newRegret
            epsilon = newEpsilon
            x = newX
            if regret < minRegret:
                minRegret, minEpsilon, minX = regret, epsilon, x
        temperature *= decreasing

    print(f"minimal regret: {minRegret}")
    print(f"epsilon: {minEpsilon}, x: {minX}")

simultedAnnealing()




