
import numpy as np
import random

ACTIONS = [
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0)
]

def train_agent(
    episodes=300,
    grid_size=8,
    obstacle_probability=0.12
):
    """
    Experimental grid-navigation Q-learning.
    This is not a flight-control neural network.
    """

    episodes = max(1, min(int(episodes), 1000))
    grid_size = max(4, min(int(grid_size), 12))

    q_table = np.zeros((
        grid_size,
        grid_size,
        len(ACTIONS)
    ))

    alpha = 0.15
    gamma = 0.95
    epsilon = 0.25

    rewards = []

    for episode in range(episodes):

        start = (0, 0)
        goal = (grid_size - 1, grid_size - 1)

        obstacles = set()

        for x in range(grid_size):
            for y in range(grid_size):
                if (x, y) not in (start, goal):
                    if random.random() < obstacle_probability:
                        obstacles.add((x, y))

        position = start
        total_reward = 0

        for step in range(150):

            x, y = position

            if random.random() < epsilon:
                action = random.randrange(4)
            else:
                action = int(np.argmax(q_table[x, y]))

            dx, dy = ACTIONS[action]

            nx = max(0, min(grid_size - 1, x + dx))
            ny = max(0, min(grid_size - 1, y + dy))

            next_position = (nx, ny)

            reward = -1

            if next_position in obstacles:
                reward = -15
                next_position = position

            elif next_position == goal:
                reward = 100

            px, py = next_position

            old_value = q_table[x, y, action]

            new_value = (
                old_value
                + alpha * (
                    reward
                    + gamma * np.max(q_table[px, py])
                    - old_value
                )
            )

            q_table[x, y, action] = new_value

            position = next_position
            total_reward += reward

            if position == goal:
                break

        rewards.append(total_reward)

    return q_table, rewards
