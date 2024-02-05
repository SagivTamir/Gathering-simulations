import pygame
import random
import math

# Define constants
WIDTH = 800
HEIGHT = 600
NUM_AGENTS = 20
AGENT_SIZE = 10
BG_COLOR = (255, 255, 255)
AGENT_COLOR = (0, 0, 255)
TARGET_COLOR = (255, 0, 0)
MAX_SPEED = 2
TARGET_RADIUS = AGENT_SIZE
GATHERING_RADIUS = 20
SIGMA = -0.02



# Define Agent class
class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dx = 0  # random.uniform(-MAX_SPEED, MAX_SPEED)
        self.dy = 0  # random.uniform(-MAX_SPEED, MAX_SPEED)

    def move_towards(self, target_x, target_y):
        # Calculate distance to target
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.sqrt(dx * dx + dy * dy)

        # If the agent is within the gathering radius, don't move
        if distance < GATHERING_RADIUS:
            return

        # Calculate normalized direction towards target
        dx /= distance
        dy /= distance

        # Adjust velocity towards the target
        self.dx += dx
        self.dy += dy

        # Normalize velocity if it exceeds maximum speed
        speed = math.sqrt(self.dx * self.dx + self.dy * self.dy)
        if speed > MAX_SPEED:
            self.dx = (self.dx / speed) * MAX_SPEED
            self.dy = (self.dy / speed) * MAX_SPEED

    def move_positional(self, locations):
        x = 0
        y = 0
        for location in locations:
            x += (self.x - location.x)
            y += (self.y - location.y)
        self.dx = SIGMA * x
        self.dy = SIGMA * y

    def move_with_bearing_only_unlimited_visibility(self, locations):
        x = 0
        y = 0
        for location in locations:
            norm = math.sqrt(((location.x-self.x) ** 2) + ((location.y-self.y) ** 2))
            if norm == 0:
                pass
            else:
                x += (self.x-location.x)/norm
                y += (self.y-location.y)/norm
        self.dx = SIGMA * x
        self.dy = SIGMA * y

    def move(self):
        self.x += self.dx
        self.y += self.dy


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agent Gathering Simulator")
clock = pygame.time.Clock()

# Create agents
agents = [Agent(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for n in range(NUM_AGENTS)]

# Main loop
running = True
while running:
    screen.fill(BG_COLOR)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Find the centroid of agents
    centroid_x = sum(agent.x for agent in agents) / NUM_AGENTS
    centroid_y = sum(agent.y for agent in agents) / NUM_AGENTS

    # Move agents towards the centroid
    for agent in agents:
        # agent.move_towards(centroid_x, centroid_y)
        # agent.move_positional(agents)
        agent.move_with_bearing_only_unlimited_visibility(agents)
        agent.move()

        # Make agents bounce off walls
        if agent.x < 0 or agent.x > WIDTH:
            agent.dx *= -1
        if agent.y < 0 or agent.y > HEIGHT:
            agent.dy *= -1

    # Draw agents
    for agent in agents:
        pygame.draw.circle(screen, AGENT_COLOR, (int(agent.x), int(agent.y)), AGENT_SIZE)

    # Draw target point
    pygame.draw.circle(screen, TARGET_COLOR, (int(centroid_x), int(centroid_y)), TARGET_RADIUS)

    pygame.display.flip()
    clock.tick(80)

pygame.quit()