import numpy as np

def UCB(delta, arms, n, theta_star):
    Mu = arms @ theta_star
    K = len(Mu)
    mu_star = np.max(Mu)

    regrets = []

    counts = np.zeros(K)
    means = np.zeros(K)

    for i in range(K):
        reward = np.random.normal(Mu[i], 1)
        counts[i] += 1
        means[i] = reward
        
    for _ in range(K, n):
        ucb_values = means + (np.sqrt((2 * np.log((2 * n * K) / delta)) / counts))
        arm = np.argmax(ucb_values)
        reward = np.random.normal(Mu[arm], 1)
        counts[arm] += 1
        means[arm] += (reward - means[arm]) / counts[arm]
        
        # Compute Regret
        regrets.append(mu_star - Mu[arm])

    return regrets