import numpy as np
import pathlib
import model
from matplotlib import pyplot as plt
import logging
import click


def get_balls(radius=0.1):
    return [
        model.Ball(x=0.25, y=0.25, vx=1.0, vy=1.0, radius=radius, name="ball1"),
        model.Ball(x=0.25, y=0.75, vx=-1.0, vy=-1.0, radius=radius, name="ball2"),
        model.Ball(x=0.75, y=0.25, vx=-1.0, vy=1.0, radius=radius, name="ball3"),
        model.Ball(x=0.75, y=0.75, vx=1.0, vy=-1.0, radius=radius, name="ball4"),
    ]


def run_single_model(radius, size, dtstore, steps, save_to_json=False, is_greedy=False):
    main_model = model.Model(
        get_balls(radius), size=size, dtstore=dtstore, is_narrow=not is_greedy
    )
    main_model.run(steps=steps)
    if save_to_json:
        main_model.save_to_json(
            f"output\model_{radius:.2f}_{steps:.0e}{'.narrow' if not is_greedy else ''}.json",
        )
        return
    plot_single_model_data(main_model.data)


def plot_single_model_data(data):
    logging.getLogger().setLevel(logging.INFO)
    plot_location_heatmap(data)
    plot_absolute_velocity_distribution(data)
    plot_x_velocity_distribution(data)
    plot_y_velocity_distribution(data)


def plot_multiple_models(radius_to_data):
    logging.getLogger().setLevel(logging.INFO)
    plot_first_quarter_probability(radius_to_data)
    plot_xvelocity_probability_to_radius(radius_to_data, separately=False)
    plot_xvelocity_probability_to_radius(radius_to_data, separately=True)


def plot_xvelocity_probability_to_radius(radius_to_data, separately=False):
    if not separately:
        plt.title(f"Ball 2 X Velocity Probability by Radius")

    for radius in radius_to_data:
        plt.xlabel("X Velocity (max abs velocity/100)", fontsize=14)
        plt.ylabel("Probability", fontsize=14)
        if separately:
            plt.title(f"Ball 2 X Velocity Probability Radius={radius:.2f}")

        distribution = radius_to_data[radius].x_velocity_distribution
        plt.plot(
            [float(x[0]) for x in sorted(distribution.items())],
            [float(x[1]) for x in sorted(distribution.items())],
            label=f"Radius = {radius:.2f}",
        )
        if separately:
            plt.show()

    if not separately:
        plt.legend(fontsize=12, loc="upper right")
        plt.show()


def plot_first_quarter_probability(radius_to_data):
    plt.title(f"Probability of Ball 2 in First Quarter")
    plt.xlabel("Radius", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.plot(
        [x[0] for x in sorted(radius_to_data.items())],
        [x[1].first_quarter_probability for x in sorted(radius_to_data.items())],
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


def plot_location_heatmap(model_data):
    plt.title("Ball 1 Heat Map")
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.imshow(model_data.get_heatmap(0), cmap="hot", interpolation="nearest")

    plt.colorbar()
    plt.show()


@click.command()
@click.argument(
    "radius", type=float, default=0.15,
)
@click.argument(
    "steps", type=float, default=2,
)
@click.option(
    "--json", type=str, default=None,
)
@click.option(
    "--greedy", default=False, is_flag=True,
)
@click.option("--json-dir", type=str, default=None)
def main(radius, steps, json, json_dir, greedy):
    if steps > 10:
        print("Warning: steps > 10, remember that steps is in log10")
        return

    if json_dir is not None:
        jsons = pathlib.Path(json_dir).glob("*.narrow.json")
        radius_to_data = {}
        for json in jsons:
            model_data = model.NarrowModelData.from_json_file(json)
            radius_to_data[model_data.radius] = model_data

        plot_multiple_models(radius_to_data)

    elif json is not None:
        model_data = model.ModelData.from_json_file(json)
        plot_single_model_data(model_data)
    else:
        run_single_model(
            radius=radius,
            size=1.0,
            dtstore=0.01,
            steps=10 ** steps,
            save_to_json=True,
            is_greedy=greedy,
        )


if __name__ == "__main__":
    main()
