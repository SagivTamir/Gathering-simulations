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
GATHERING_RADIUS = 100
SIGMA = 0.02

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agent Gathering Simulator")
clock = pygame.time.Clock()

# Initialize agent positions and velocities using NumPy arrays
positions = np.float64(np.random.randint(0, high=[WIDTH, HEIGHT], size=(NUM_AGENTS, 2)))
velocities = np.zeros((NUM_AGENTS, 2))

def move_agents(positions, velocities):
    centroid = positions.mean(axis=0)
    directions = centroid - positions
    distances = np.linalg.norm(directions, axis=1, keepdims=True)
    valid = distances[:, 0] >= GATHERING_RADIUS
    normalized_directions = np.divide(directions, distances, where=valid[:, None])
    velocities[valid] += SIGMA * normalized_directions[valid]
    
    # Normalize velocities to have a maximum speed
    speeds = np.linalg.norm(velocities, axis=1)
    too_fast = speeds > MAX_SPEED
    # Create a mask for elements that need to be scaled
    scale_factors = MAX_SPEED / speeds[too_fast]
    # Apply the mask and scale velocities
    velocities[too_fast] = velocities[too_fast] * scale_factors[:, np.newaxis]

    # Update positions
    positions += velocities
    
    # Bounce off walls
    positions[:, 0] = np.clip(positions[:, 0], 0, WIDTH)
    positions[:, 1] = np.clip(positions[:, 1], 0, HEIGHT)
    bounce_mask_x = (positions[:, 0] == 0) | (positions[:, 0] == WIDTH)
    bounce_mask_y = (positions[:, 1] == 0) | (positions[:, 1] == HEIGHT)
    velocities[bounce_mask_x, 0] *= -1
    velocities[bounce_mask_y, 1] *= -1

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
