import pygame
import random
import math

# Define constants
WIDTH = 1000
HEIGHT = 1000
NUM_AGENTS = 200
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

    def move_positional(self, agents):
        x = 0
        y = 0
        for agent in agents:
            if agent != self:
                x += self.x - agent.x
                y += self.y - agent.y
        self.dx = SIGMA * (x / len(agents))
        self.dy = SIGMA * (y / len(agents))

    def move_with_bearing_only(self, agents):
        x = 0
        y = 0
        for agent in agents:
            if agent != self:
                norm = math.sqrt(
                    ((agent.x - self.x) ** 2) + ((agent.y - self.y) ** 2)
                )
                if norm == 0:
                    pass
                else:
                    x += (self.x - agent.x) / norm
                    y += (self.y - agent.y) / norm
        self.dx = SIGMA * x 
        self.dy = SIGMA * y
        print(x**2 + y**2)

    def move(self):
        self.x += self.dx
        self.y += self.dy


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agent Gathering Simulator")
clock = pygame.time.Clock()

# Create agents
agents = [
    Agent(random.randint(0, WIDTH), random.randint(0, HEIGHT))
    for n in range(NUM_AGENTS)
]

# Main loop
running = True
i = 0
while running:
    screen.fill(BG_COLOR)

    i = (i + 1) % NUM_AGENTS

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Find the centroid of agents
    centroid_x = sum(agent.x for agent in agents) / NUM_AGENTS
    centroid_y = sum(agent.y for agent in agents) / NUM_AGENTS

    # Move agents towards the centroid
    for agent_i, agent in enumerate(agents):
        # Simulate continuous time by only updating movement every so often and in random order
        if agent_i % 50 == i % 50:
            # agent.move_towards(centroid_x, centroid_y)
            # agent.move_positional(agents)
            agent.move_with_bearing_only(agents)
        agent.move()

    # Draw agents
    for agent in agents:
        pygame.draw.circle(
            screen, AGENT_COLOR, (int(agent.x), int(agent.y)), AGENT_SIZE
        )

    # Draw target point
    pygame.draw.circle(
        screen, TARGET_COLOR, (int(centroid_x), int(centroid_y)), TARGET_RADIUS
    )

    pygame.display.flip()
    clock.tick(100)

pygame.quit()
