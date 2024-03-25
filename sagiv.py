import pygame
import random
import math
import numpy as np

# Define constants
COLLISIONS = 0
WIDTH = 1000
HEIGHT = 800
NUM_AGENTS = 10
AGENT_SIZE = 10
BG_COLOR = (255, 255, 255)
AGENT_COLOR = (0, 0, 255)
TARGET_COLOR = (255, 0, 0)
MAX_SPEED = 3
TARGET_RADIUS = AGENT_SIZE
GATHERING_RADIUS = 20


NUM_AGENTS = 3
POSITION_SENSING = 0
BEARING_ONLY = 1
STUDY_CASE = BEARING_ONLY

SIGMA = -0.2 if STUDY_CASE == BEARING_ONLY else -0.001

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
                x += self.p[0] - agent.p[0]
                y += self.p[1] - agent.p[1]
        self.dp[0] = SIGMA * (x / NUM_AGENTS)
        self.dp[1] = SIGMA * (y / NUM_AGENTS)

    def move_with_bearing_only(self, agents):
        x = 0
        y = 0
        for agent in agents:
            if agent != self:
                norm = math.sqrt(((agent.p[0] - self.p[0]) ** 2) + ((agent.p[1] - self.p[1]) ** 2))
                if norm == 0:
                    pass
                else:
                    x += (self.p[0] - agent.p[0]) / norm
                    y += (self.p[1] - agent.p[1]) / norm
        self.dp[0] = SIGMA * x
        self.dp[1] = SIGMA * y

    def move(self, agents):
        if not COLLISIONS:
            self.p += self.dp
            return

        collisions = []
        for agent in agents:
            if agent != self:
                distance = np.linalg.norm(self.p - agent.p)
                if distance <= 2 * AGENT_SIZE:
                    collisions += [(self.p - agent.p) / distance]
                    # return

        if not collisions:
            self.p += self.dp
            return

        v_dp = np.linalg.norm(self.dp)
        u_dp = self.dp / v_dp

        for collision in collisions:
            # Calculate the component of the collision normal perpendicular to the velocity
            collision_perpendicular = collision - np.dot(u_dp, collision) * u_dp
            collision_perpendicular_norm = np.linalg.norm(collision_perpendicular)
            if collision_perpendicular_norm > 0:
                collision_perpendicular /= collision_perpendicular_norm
            u_dp -= np.dot(u_dp, collision_perpendicular) * collision_perpendicular

        u_dp_norm = np.linalg.norm(u_dp)
        if u_dp_norm > 0:
            self.dp = u_dp / u_dp_norm * v_dp
        else:
            return

        self.p += self.dp



def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Agent Gathering Simulator")
    clock = pygame.time.Clock()

    # Create agents
    agents = [
        Agent(
            np.array([random.randint(0, WIDTH), random.randint(0, HEIGHT)], dtype=float)
        )
        for _ in range(NUM_AGENTS)
    ]

    # Initialize font
    font = pygame.font.SysFont('Arial', 30)

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
            # # Simulate continuous time by only updating movement every so often and in random order
            # if agent_i % 20 == i % 20:
            #     agent.move_towards(centroid_p)
            if (STUDY_CASE == POSITION_SENSING):
                agent.move_positional(agents)
            if (STUDY_CASE == BEARING_ONLY):
                agent.move_with_bearing_only(agents)

            agent.move(agents)

        # Draw agents
        for agent in agents:
            pygame.draw.circle(
                screen, AGENT_COLOR, (int(agent.p[0]), int(agent.p[1])), AGENT_SIZE
            )

        # Draw target point
        pygame.draw.circle(
            screen,
            TARGET_COLOR,
            (int(centroid_p[0]), int(centroid_p[1])),
            TARGET_RADIUS,
        )

        algorithm_names = ["Positional Sensing", "Bearing Only"]
        text = f"Algorithm: {algorithm_names[STUDY_CASE]}, Agents: {NUM_AGENTS}, Sigma: {-SIGMA}"
        text_surface = font.render(text, True, (0, 0, 0))
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        clock.tick(100)

    pygame.quit()


if __name__ == "__main__":
    main()
