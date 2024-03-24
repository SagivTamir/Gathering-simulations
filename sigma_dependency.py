import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_agents = 8
sigmas = [1/n_agents,2/n_agents, 3/n_agents]  # Different sigma values to explore
n_steps = 10


def update_states(states, sigma):
    n = len(states)
    new_states = np.zeros(n)
    for i in range(n):
        diff_sum = np.sum(states[i] - states)  # Sum of differences with all other agents
        new_states[i] = states[i] - sigma * diff_sum
    return new_states

def run_dynamics(n_agents, sigma, n_steps=100, initial_states=None):
    if initial_states is None:
        states = np.random.rand(n_agents)  # Initialize with random states
    else:
        states = np.array(initial_states)
    average = np.average(states)

    print(average)

    history = [states.copy()]
    for _ in range(n_steps):
        states = update_states(states, sigma)
        history.append(states.copy())
    return np.array(history)



# Run the dynamics for each sigma and plot separately
for sigma_index, sigma in enumerate(sigmas):
    plt.figure(figsize=(10, 4))
    history = run_dynamics(n_agents, sigma, n_steps)

    # Plotting the state of each agent over time for the current sigma
    for i in range(n_agents):
        plt.plot(history[:, i], label=f'Agent {i+1}')

    plt.xlabel('Time step')
    plt.ylabel('State')
    plt.title(f'Discrete Time Dynamics of Agents with Sigma = {sigma}')
    plt.legend()
    plt.show()
