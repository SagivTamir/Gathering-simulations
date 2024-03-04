import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import ConvexHull, distance
from tqdm import tqdm

# Parameters
n = 10  # Number of agents
nb_steps = 1000  # Number of steps
dt = 1
v = 1  # Speed: Step size of v*dt
square_size = 50  # Size of the square
nb_trials = 1  # Number of runs, reduced for demonstration
limited_range = None  # Limited range, None if none
compute_circle = True  # Compute enclosing circle
save_video = True  # Save the video


# Function to compute the smallest enclosing circle
def smallest_enclosing_circle(points):
    hull = ConvexHull(points)
    circle_center, circle_radius = hull.points.mean(axis=0), np.max(
        distance.cdist([hull.points.mean(axis=0)], hull.points)
    )
    return circle_center, circle_radius


# Define the update function for the animation
def update(frame):
    global p, circle, ax
    alpha = np.random.rand(n) * 2 * np.pi
    p_next = p.copy()

    for i in range(n):
        dir_i = p[:, i].reshape(2, 1) - np.delete(p, i, axis=1)
        dists_i = np.linalg.norm(dir_i, axis=0)
        u_i = dir_i / dists_i

        u_alpha = np.array([np.cos(alpha[i]), np.sin(alpha[i])])
        v_i = -u_alpha * v

        scal_i = np.sum(u_alpha.reshape(2, 1) * u_i, axis=0)

        visible_i = (
            np.logical_and(scal_i <= 0, dists_i <= limited_range)
            if limited_range is not None
            else scal_i <= 0
        )

        if np.sum(visible_i) > 0:
            v_i = np.array([0, 0])

        p_next[:, i] = p[:, i] + v_i * dt

    p = p_next

    if compute_circle:
        circle_center, circle_radius = smallest_enclosing_circle(p.T)
        circle.set_radius(circle_radius)
        circle.set_center(circle_center)

    ax.clear()
    ax.scatter(p[0], p[1])
    if compute_circle:
        ax.add_artist(circle)
    ax.set_xlim([0, square_size])
    ax.set_ylim([0, square_size])


# Simulation loop
for trial in tqdm(range(nb_trials)):
    p = np.random.rand(2, n) * square_size  # Initial positions

    fig, ax = plt.subplots()
    circle = plt.Circle((0, 0), 1, color="r", fill=False)  # Placeholder circle
    if compute_circle:
        ax.add_artist(circle)
    ax.set_xlim([0, square_size])
    ax.set_ylim([0, square_size])

    # Create the FuncAnimation
    anim = FuncAnimation(fig, update, frames=nb_steps, repeat=False)

    # Save the animation
    if save_video:
        video_file = f"ants_trial_{trial}.gif"
        anim.save(video_file, writer=PillowWriter(fps=50))  # Adjust fps as needed
