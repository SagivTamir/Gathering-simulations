import pygame
import numpy as np
import time
from functools import wraps
from scipy.spatial import cKDTree

# Define constants
WIDTH = 1000
HEIGHT = 700
NUM_AGENTS = 50
AGENT_SIZE = 10
BG_COLOR = (255, 255, 255)
AGENT_COLOR = (0, 0, 255)
TARGET_COLOR = (255, 0, 0)
MAX_SPEED = 2
TARGET_RADIUS = AGENT_SIZE
GATHERING_RADIUS = 20
SIGMA = 0.1
ALLOW_INITIAL_OVERLAPS = True
ALLOW_COLLISIONS = False
COLLISION_ALGORITHM = 0  # 0 = Naive, 1 = Sweep and Prune, 2 = KD tree

tracker_latest_execution_time = 0.0
tracker_average_execution_time = 0.0


def avg_time_tracker(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # This is where we'll keep track of all the execution times
        if not hasattr(wrapper, "execution_times"):
            wrapper.execution_times = []

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Calculate the execution time for this call and store it
        execution_time = end_time - start_time
        wrapper.execution_times.append(execution_time)

        # Calculate the average execution time
        average_execution_time = sum(wrapper.execution_times) / len(
            wrapper.execution_times
        )
        print(
            f"Execution time of {func.__name__}: Current {execution_time:.5f}, Average: {average_execution_time:.5f}"
        )
        global tracker_average_execution_time, tracker_latest_execution_time
        tracker_average_execution_time = average_execution_time
        tracker_latest_execution_time = execution_time

        return result

    return wrapper


def initialize_positions(num_agents, agent_size, width, height):
    if ALLOW_INITIAL_OVERLAPS:
        return np.float64(
            np.random.randint(0, high=[WIDTH, HEIGHT], size=(NUM_AGENTS, 2))
        )

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


@avg_time_tracker
def vectorized_collision_adjustment_kd_tree(positions, velocities):
    # Create a k-d tree for positions
    tree = cKDTree(positions)

    # Query the k-d tree for neighbors within collision distance
    collision_pairs = tree.query_pairs(r=2 * AGENT_SIZE)

    # Initialize an empty list to store collision adjustments for each agent
    velocity_adjustments = np.zeros_like(velocities)

    for i, j in collision_pairs:
        # Calculate the vector from agent i to agent j
        collision_vector = positions[j] - positions[i]
        distance = np.linalg.norm(collision_vector)

        # Avoid division by zero
        if distance == 0:
            continue

        normalized_collision_vector = collision_vector / distance
        dot_product_i = np.dot(velocities[i], normalized_collision_vector)
        dot_product_j = np.dot(velocities[j], normalized_collision_vector)

        # Adjust velocity for agent i if moving towards agent j
        if dot_product_i > 0:
            velocity_adjustments[i] -= dot_product_i * normalized_collision_vector

        # Adjust velocity for agent j if moving towards agent i
        if dot_product_j < 0:
            velocity_adjustments[j] -= dot_product_j * normalized_collision_vector

    # Apply velocity adjustments
    velocities += velocity_adjustments


@avg_time_tracker
def sweep_and_prune_collision_adjustment(positions, velocities):
    def find_potential_pairs(sorted_indices, axis):
        potentials = set()
        for i in range(NUM_AGENTS - 1):
            for j in range(i + 1, NUM_AGENTS):
                if (
                    positions[sorted_indices[j], axis]
                    - positions[sorted_indices[i], axis]
                    < 2 * AGENT_SIZE
                ):
                    potentials.add(
                        (
                            min(sorted_indices[i], sorted_indices[j]),
                            max(sorted_indices[i], sorted_indices[j]),
                        )
                    )
                else:
                    break
        return potentials

    potential_pairs_x = find_potential_pairs(np.argsort(positions[:, 0]), 0)
    # potential_pairs_y = find_potential_pairs(np.argsort(positions[:, 1]), 1)

    # Intersection of potential pairs in both axes
    # potential_pairs = list(potential_pairs_x.intersection(potential_pairs_y))
    potential_pairs = list(potential_pairs_x)

    # Collision Detection
    if potential_pairs:
        pair_indices = np.array(potential_pairs)
        p1_indices, p2_indices = pair_indices[:, 0], pair_indices[:, 1]

        p1_positions = positions[p1_indices]
        p2_positions = positions[p2_indices]
        diff_positions = p1_positions - p2_positions
        distances = np.linalg.norm(diff_positions, axis=1)

        collision_mask = (distances > 0) & (distances < 2 * AGENT_SIZE)
        collision_indices = np.where(collision_mask)[0]  # Indices of pairs that collide

        # Collision Resolution
        for index in collision_indices:
            i, j = p1_indices[index], p2_indices[index]
            collision_vector = diff_positions[index]
            distance = distances[index]
            normalized_collision_vector = collision_vector / distance

            dot_product_i = np.dot(velocities[i], normalized_collision_vector)
            if dot_product_i < 0:
                velocities[i] -= dot_product_i * normalized_collision_vector

            dot_product_j = np.dot(velocities[j], -normalized_collision_vector)
            if dot_product_j < 0:
                velocities[j] += dot_product_j * normalized_collision_vector


@avg_time_tracker
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
        if COLLISION_ALGORITHM == 0:
            vectorized_collision_adjustment(positions, velocities)
        elif COLLISION_ALGORITHM == 1:
            sweep_and_prune_collision_adjustment(positions, velocities)
        else:
            vectorized_collision_adjustment_kd_tree(positions, velocities)

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

    # Initialize font
    font = pygame.font.SysFont('Arial', 24)

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

        algorithm_names = ["Naive", "Sweep and Prune", "KD Tree"]
        global tracker_latest_execution_time, tracker_average_execution_time
        text = f"Algorithm: {algorithm_names[COLLISION_ALGORITHM]}, Agents: {NUM_AGENTS}, Latest: {tracker_latest_execution_time:.5f}s, Avg: {tracker_average_execution_time:.5f}s"
        text_surface = font.render(text, True, (0, 0, 0))
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
