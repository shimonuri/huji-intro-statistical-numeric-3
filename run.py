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


def main(radius, size, dtstore, steps):
    main_model = model.Model(get_balls(radius), size=size, dtstore=dtstore)
    main_model.run(steps=steps)
    logging.basicConfig(level=logging.INFO)
    plot_location_heatmap(main_model.data)

def plot_locations(main_model):
    plt.plot(main_model.data.get_x(0), main_model.data.get_y(0), label="ball1")
    plt.plot(main_model.data.get_x(1), main_model.data.get_y(1), label="ball2")
    plt.plot(main_model.data.get_x(2), main_model.data.get_y(2), label="ball3")
    plt.plot(main_model.data.get_x(3), main_model.data.get_y(3), label="ball4")
    plt.legend()
    plt.show()


def plot_location_heatmap(model_data):
    plt.imshow(model_data.get_heatmap(2), cmap="hot", interpolation="nearest")

    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main(radius=0.05, size=1.0, dtstore=0.01, steps=10000)
