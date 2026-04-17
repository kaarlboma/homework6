import numpy as np

def LinearBandit(d, lam, action_set, n, theta_star, delta, L = 1, S = 1):
    # Initialization
    sigma = lam * np.eye(d)
    theta_hat = np.zeros(d)
    b = np.zeros(d)
    regrets = []
    beta = ((np.sqrt(lam) * S) + np.sqrt(np.log(1 / delta) + (d * np.log(1 + ((n * (L ** 2)) / (lam * d)))))) ** 2
    for t in range(n):
        # Observe action set and play action
        sigma_inv = np.linalg.inv(sigma)

        if action_set is None:
            a_t = select_action(d, theta_hat, sigma_inv, beta)
        else:
            scores = np.zeros(len(action_set))
            for i in range(len(action_set)):
                a = action_set[i]
                scores[i] = a @ theta_hat + np.sqrt(beta) * np.sqrt(a @ sigma_inv @ a)
            a_t = action_set[np.argmax(scores)]

        r_t = (theta_star @ a_t) + np.random.randn()

        # Update
        sigma = sigma + np.outer(a_t, a_t)
        b = b + (a_t * r_t)
        theta_hat = np.linalg.inv(sigma) @ b

        if action_set is None:
            regret = np.linalg.norm(theta_star) - theta_star @ a_t
            regrets.append(regret)
        else:
            regret = np.max([theta_star @ a for a in action_set]) - theta_star @ a_t
            regrets.append(regret)

    return regrets

def select_action(d, theta_hat, sigma_inv, beta, num_iter=1000, lr=0.01):
    a = np.random.randn(d)
    for _ in range(num_iter):
        a = a / np.linalg.norm(a)
        g = theta_hat + (np.sqrt(beta) * (sigma_inv @ a) / np.sqrt(a @ sigma_inv @ a))
        a = a + (lr * g)
        a = a / np.linalg.norm(a)
    return a