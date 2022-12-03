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


def main(radius, size, dtstore):
    logging.basicConfig(level=logging.INFO)
    main_model = model.Model(get_balls(radius), size=size, dtstore=dtstore)
    main_model.run()
    plt.plot(main_model.data.times, main_model.data.get_x(0))
    plt.show()


if __name__ == "__main__":
    main(radius=0.1, size=1.0, dtstore=0.01)
