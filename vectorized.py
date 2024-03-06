import pygame
import numpy as np
import random

# Define constants
WIDTH = 1000
HEIGHT = 1000
NUM_AGENTS = 200
AGENT_SIZE = 10
BG_COLOR = (255, 255, 255)
AGENT_COLOR = (0, 0, 255)
TARGET_COLOR = (255, 0, 0)
MAX_SPEED = 1
TARGET_RADIUS = AGENT_SIZE
GATHERING_RADIUS = 20
SIGMA = 0.02
ALLOW_COLLISIONS = True
ALLOW_INITIAL_OVERLAPS = False


def initialize_positions(num_agents, agent_size, width, height):
    if ALLOW_INITIAL_OVERLAPS:
        return np.float64(np.random.randint(0, high=[WIDTH, HEIGHT], size=(NUM_AGENTS, 2)))

    positions = np.zeros((num_agents, 2), dtype=np.float64)
    for i in range(num_agents):
        while True:
            # Generate a random position for the new agent
            new_pos = np.random.randint(
                low=agent_size,
                high=[width - agent_size, height - agent_size],
                size=(1, 2),
            )
            # Calculate distances from the new agent to all existing agents
            if i == 0:
                # If it's the first agent, just break the loop as there are no other agents to check distance from
                positions[i] = new_pos
                break
            else:
                distances = np.linalg.norm(positions[:i] - new_pos, axis=1)
                # Check if the new position is at least 2 * AGENT_SIZE away from all existing agents
                if np.all(distances >= 2.1 * agent_size):
                    positions[i] = new_pos
                    break
    return positions


def vectorized_collision_adjustment(positions, velocities):
    # Detect and resolve collisions
    diff_positions = positions[:, np.newaxis] - positions
    distances_matrix = np.linalg.norm(diff_positions, axis=2)
    collision_mask = (0 < distances_matrix) & (distances_matrix < 2 * AGENT_SIZE)

    for i in range(NUM_AGENTS):
        colliders = np.where(collision_mask[i])[0]
        for collider in colliders:
            # Vector from i to collider
            collision_vector = diff_positions[i, collider]
            distance = distances_matrix[i, collider]

            if distance == 0:  # Avoid division by zero
                continue

            normalized_collision_vector = collision_vector / distance
            dot_product_i = np.dot(velocities[i], normalized_collision_vector)

            # Adjust velocity for agent i if moving towards collider
            if dot_product_i < 0:
                velocities[i] -= dot_product_i * normalized_collision_vector


def move_agents(positions, velocities):
    centroid = positions.mean(axis=0)
    directions = centroid - positions
    distances = np.linalg.norm(directions, axis=1)
    # valid = distances >= GATHERING_RADIUS
    # normalized_directions = np.divide(directions, distances.reshape(-1, 1), where=valid.reshape(-1, 1))
    # velocities[valid] += SIGMA * normalized_directions[valid]
    normalized_directions = np.divide(directions, distances.reshape(-1, 1))
    velocities += SIGMA * normalized_directions

    if not ALLOW_COLLISIONS:
        # Collision adjustment
        vectorized_collision_adjustment(positions, velocities)

    # Normalize velocities to have a maximum speed
    speeds = np.linalg.norm(velocities, axis=1)
    too_fast = speeds > MAX_SPEED
    scale_factors = MAX_SPEED / speeds[too_fast]
    velocities[too_fast] = velocities[too_fast] * scale_factors.reshape(-1, 1)

    # Update positions
    positions += velocities


def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Agent Gathering Simulator")
    clock = pygame.time.Clock()

    # Initialize agent positions and velocities using NumPy arrays
    positions = initialize_positions(NUM_AGENTS, AGENT_SIZE, WIDTH, HEIGHT)
    velocities = np.zeros((NUM_AGENTS, 2))

    # Main loop
    running = True
    while running:
        screen.fill(BG_COLOR)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Move agents
        move_agents(positions, velocities)

        # Draw agents
        for pos in positions:
            pygame.draw.circle(screen, AGENT_COLOR, pos.astype(int), AGENT_SIZE)

        # Draw target point
        centroid = positions.mean(axis=0)
        pygame.draw.circle(screen, TARGET_COLOR, centroid.astype(int), TARGET_RADIUS)

        pygame.display.flip()
        clock.tick(100)

    pygame.quit()


if __name__ == "__main__":
    main()
