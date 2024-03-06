import pygame
import random
import math
import numpy as np

# Define constants
WIDTH = 1000
HEIGHT = 1000
NUM_AGENTS = 200
AGENT_SIZE = 10
BG_COLOR = (255, 255, 255)
AGENT_COLOR = (0, 0, 255)
TARGET_COLOR = (255, 0, 0)
MAX_SPEED = 5
TARGET_RADIUS = AGENT_SIZE
GATHERING_RADIUS = 20
SIGMA = -0.02
COLLISIONS = True

# Define Agent class
class Agent:
    def __init__(self, p):
        self.p = p
        self.dp = np.zeros(2)

    def move_towards(self, target_p):
        # Calculate distance to target
        dp = -(self.p - target_p)
        distance = np.linalg.norm(dp)

        # If the agent is within the gathering radius, don't move
        if distance < GATHERING_RADIUS:
            return

        # Calculate normalized direction towards target
        dp /= distance

        # Adjust velocity towards the target
        self.dp += dp

        # Normalize velocity if it exceeds maximum speed
        speed = np.linalg.norm(self.dp)
        if speed > MAX_SPEED:
            self.dp = (self.dp / speed) * MAX_SPEED

    def move_positional(self, agents):
        x = 0
        y = 0
        for agent in agents:
            if agent != self:
                x += self.x - agent.x
                y += self.y - agent.y
        self.dx = SIGMA * (x / NUM_AGENTS)
        self.dy = SIGMA * (y / NUM_AGENTS)

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

    # def move(self, agents):
    #     if not COLLISIONS:
    #         self.x += self.dx
    #         self.y += self.dy
    #         return
        
    #     colliding_agents = []
    #     for agent in agents:
    #         if agent != self:
    #             distance =  math.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2)
    #             if distance <= 2 * AGENT_SIZE:
    #                 colliding_agents += [agent]
    #                 return
        
    #     unit
    #     self.x += self.dx
    #     self.y += self.dy

    def move(self, agents):
        if not COLLISIONS:
            self.p += self.dp
            return
        
        colliding_agents = []
        for agent in agents:
            if agent != self:
                distance =  np.linalg.norm(self.p - agent.p)
                if distance <= 2 * AGENT_SIZE:
                    colliding_agents += [(self.p - agent.p) / distance]
                    return
        
        u_dp = self.dp / np.linalg.norm(self.dp)

        self.p += self.dp


def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Agent Gathering Simulator")
    clock = pygame.time.Clock()

    # Create agents
    agents = [
        Agent(np.array([random.randint(0, WIDTH), random.randint(0, HEIGHT)], dtype=float))
        for _ in range(NUM_AGENTS)
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
        centroid_p = sum(agent.p for agent in agents) / NUM_AGENTS

        # Move agents towards the centroid
        for agent_i, agent in enumerate(agents):
            # Simulate continuous time by only updating movement every so often and in random order
            # if agent_i % 50 == i % 50:
            agent.move_towards(centroid_p)
                # agent.move_positional(agents)
            # agent.move_with_bearing_only(agents)
            agent.move(agents)

        # Draw agents
        for agent in agents:
            pygame.draw.circle(
                screen, AGENT_COLOR, (int(agent.p[0]), int(agent.p[1])), AGENT_SIZE
            )

        # Draw target point
        pygame.draw.circle(
            screen, TARGET_COLOR, (int(centroid_p[0]), int(centroid_p[1])), TARGET_RADIUS
        )

        pygame.display.flip()
        clock.tick(100)

    pygame.quit()


if __name__ == "__main__":
    main()
