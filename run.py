import numpy as np

import model
from matplotlib import pyplot as plt
import logging


def get_balls(radius=0.1):
    return [
        model.Ball(x=0.25, y=0.25, vx=1.0, vy=1.0, radius=radius, name="ball1"),
        model.Ball(x=0.25, y=0.75, vx=-1.0, vy=-1.0, radius=radius, name="ball2"),
        model.Ball(x=0.75, y=0.25, vx=-1.0, vy=1.0, radius=radius, name="ball3"),
        model.Ball(x=0.75, y=0.75, vx=1.0, vy=-1.0, radius=radius, name="ball4"),
    ]


def run_single_model(radius, size, dtstore, steps):
    main_model = model.Model(get_balls(radius), size=size, dtstore=dtstore)
    main_model.run(steps=steps)

    logging.getLogger().setLevel(logging.INFO)
    plot_location_heatmap(main_model.data)
    plot_absolute_velocity_distribution(main_model.data)
    plot_x_velocity_distribution(main_model.data)
    plot_y_velocity_distribution(main_model.data)


def run_multiple_models(size, dtstore, steps):
    radius_to_model = {}
    for radius in np.linspace(0.1, 0.23, 14):
        radius_model = model.Model(get_balls(radius), size=size, dtstore=dtstore)
        radius_model.run(steps=steps)
        radius_to_model[radius] = radius_model

    logging.getLogger().setLevel(logging.INFO)
    plot_first_quarter_probability(radius_to_model, 1)
    plot_xvelocity_probability_to_radius(radius_to_model, 1, separately=False)
    plot_xvelocity_probability_to_radius(radius_to_model, 1, separately=True)


def plot_xvelocity_probability_to_radius(
    radius_to_model, ball_number, separately=False
):
    if not separately:
        plt.title(f"Ball {ball_number + 1} X Velocity Probability by Radius")

    for radius in radius_to_model:
        plt.xlabel("X Velocity (max abs velocity/100)", fontsize=14)
        plt.ylabel("Probability", fontsize=14)
        if separately:
            plt.title(
                f"Ball {ball_number + 1} X Velocity Probability Radius={radius:.2f}"
            )

        distribution = radius_to_model[radius].data.get_vx(
            ball_number, is_distribution=True
        )
        plt.plot(
            [x[0] for x in sorted(distribution.items())],
            [x[1] for x in sorted(distribution.items())],
            label=f"Radius = {radius:.2f}",
        )
        if separately:
            plt.show()

    if not separately:
        plt.legend(fontsize=12, loc="upper right")
        plt.show()


def plot_first_quarter_probability(radius_to_model, ball_number):
    plt.title(f"Probability of Ball {ball_number + 1} in First Quarter")
    plt.xlabel("Radius", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.plot(
        [x[0] for x in sorted(radius_to_model.items())],
        [
            x[1].data.get_first_quarter_probability(ball_number)
            for x in sorted(radius_to_model.items())
        ],
    )
    plt.show()


def plot_x_velocity_distribution(model_data):
    plt.title("Balls X Velocity Distribution")
    plt.xlabel("X Velocity (max abs velocity/100)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    for ball_number in range(len(model_data.model_states[0].balls)):
        distribution = model_data.get_vx(ball_number, is_distribution=True)
        plt.plot(
            [x[0] for x in sorted(distribution.items())],
            [x[1] for x in sorted(distribution.items())],
            label=f"Ball {ball_number + 1}",
        )
    plt.legend()
    plt.show()


def plot_y_velocity_distribution(model_data):
    plt.title("Balls Y Velocity Distribution")
    plt.xlabel("Y Velocity (max abs velocity/100)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    for ball_number in range(len(model_data.model_states[0].balls)):
        distribution = model_data.get_vy(ball_number, is_distribution=True)
        plt.plot(
            [x[0] for x in sorted(distribution.items())],
            [x[1] for x in sorted(distribution.items())],
            label=f"Ball {ball_number + 1}",
        )
    plt.legend()
    plt.show()


def plot_absolute_velocity_distribution(model_data):
    plt.title("Balls Absolute Velocity Distribution")
    plt.xlabel("Absolute Velocity / (max abs velocity/100)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    for ball_number in range(len(model_data.model_states[0].balls)):
        distribution = model_data.get_vabs(ball_number, is_distribution=True)
        plt.plot(
            [x[0] for x in sorted(distribution.items())],
            [x[1] for x in sorted(distribution.items())],
            label=f"Ball {ball_number + 1}",
        )
    plt.legend()
    plt.show()


def plot_locations(main_model):
    plt.plot(main_model.data.get_x(0), main_model.data.get_y(0), label="ball1")
    plt.plot(main_model.data.get_x(1), main_model.data.get_y(1), label="ball2")
    plt.plot(main_model.data.get_x(2), main_model.data.get_y(2), label="ball3")
    plt.plot(main_model.data.get_x(3), main_model.data.get_y(3), label="ball4")
    plt.legend()
    plt.show()


def plot_location_heatmap(model_data):
    plt.title("Ball 1 Heat Map")
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.imshow(model_data.get_heatmap(0), cmap="hot", interpolation="nearest")

    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # run_single_model(radius=0.15, size=1.0, dtstore=0.01, steps=1e7)
    run_multiple_models(size=1.0, dtstore=0.01, steps=1e7)
