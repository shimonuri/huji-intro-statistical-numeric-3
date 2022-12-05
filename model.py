import dataclasses
import dataclasses_json
import itertools

import numpy as np
import json

import collisions
import logging
from typing import List, Dict, Tuple

# configure logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

CELL_SIZE = 0.1


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    name: str = "ball"

    def move(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

    @property
    def cell(self):
        return (int(self.x // CELL_SIZE), int(self.y // CELL_SIZE))

    @property
    def vabs(self):
        return (self.vx ** 2 + self.vy ** 2) ** 0.5

    def __str__(self):
        return f"Ball(x={self.x}, y={self.y}, vx={self.vx}, vy={self.vy}, radius={self.radius})"


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class BallsCollision:
    ball_1: Ball
    ball_2: Ball
    dt: float

    def __str__(self):
        return (
            f"BallsCollision(ball_1={self.ball_1}, ball_2={self.ball_2}, dt={self.dt})"
        )


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class WallCollision:
    ball: Ball
    wall: Tuple[int]
    dt: float

    def __str__(self):
        return f"WallCollision(ball={self.ball}, dt={self.dt})"


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class ModelState:
    balls: List[Ball]
    time: float


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class ModelData:
    model_states: List[ModelState]
    velocity_cell_size: float

    @classmethod
    def from_json_file(cls, filename):
        with open(filename, "rt") as fd:
            return cls.from_dict(json.load(fd))

    @property
    def radius(self):
        """
        assumes that all balls have the same radius
        """
        return self.model_states[0].balls[0].radius

    @property
    def times(self):
        return [state.time for state in self.model_states]

    def add_state(self, state: ModelState):
        self.model_states.append(state)

    def get_x(self, ball_number):
        return [state.balls[ball_number].x for state in self.model_states]

    def get_y(self, ball_number):
        return [state.balls[ball_number].y for state in self.model_states]

    def get_heatmap(self, ball_number):
        heatmap = np.zeros((int(1 / CELL_SIZE), int(1 / CELL_SIZE)))
        for state in self.model_states:
            heatmap[state.balls[ball_number].cell] += 1

        return heatmap / sum(sum(heatmap))

    def _get_distribution(self, values):
        return {value: values.count(value) / len(values) for value in set(values)}

    def get_first_quarter_probability(self, ball_number):
        return sum(
            [
                1
                for state in self.model_states
                if state.balls[ball_number].x <= 0.5
                and state.balls[ball_number].y <= 0.5
            ]
        ) / len(self.model_states)

    def get_vabs(self, ball_number, is_distribution=False):
        if is_distribution:
            values = [
                state.balls[ball_number].vabs // self.velocity_cell_size
                for state in self.model_states
            ]
            return self._get_distribution(values)

        return [state.balls[ball_number].vabs for state in self.model_states]

    def get_vx(self, ball_number, is_distribution=False):
        if is_distribution:
            values = [
                state.balls[ball_number].vx // self.velocity_cell_size
                for state in self.model_states
            ]
            return self._get_distribution(values)

        return [state.balls[ball_number].vx for state in self.model_states]

    def get_vy(self, ball_number, is_distribution=False):
        if is_distribution:
            values = [
                state.balls[ball_number].vy // self.velocity_cell_size
                for state in self.model_states
            ]
            return self._get_distribution(values)

        return [state.balls[ball_number].vy for state in self.model_states]


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class NarrowModelData:
    radius: float
    ball_number: int
    first_quarter_probability: float
    x_velocity_distribution: Dict[float, float]

    @classmethod
    def from_json_file(cls, filename):
        with open(filename, "rt") as fd:
            return cls.from_dict(json.load(fd))


class Model:
    def __init__(
        self,
        balls,
        size,
        dtstore,
        velocity_cell_size=None,
        ball_index_to_follow=1,
        is_narrow=False,
    ):
        if velocity_cell_size is None:
            velocity_cell_size = np.sqrt(sum([ball.vabs ** 2 for ball in balls])) / 100

        self.balls = balls
        self.size = size
        self.dtstore = dtstore
        self.time = 0
        self.is_narrow = is_narrow
        self.ball_index_to_follow = ball_index_to_follow
        self.velocity_cell_size = velocity_cell_size
        self.data = ModelData(model_states=[], velocity_cell_size=velocity_cell_size)

    def run(self, steps=100):
        step = 0
        while step <= steps:
            step += 1
            collision = self._find_next_collision()
            self._forward(collision.dt)
            if isinstance(collision, BallsCollision):
                collisions.make_balls_collision(collision.ball_1, collision.ball_2)
            elif isinstance(collision, WallCollision):
                collisions.make_wall_collision(collision.ball, collision.wall)

            if step % (steps // 100) == 0:
                logging.debug(f"step={step}, {collision}")

        return self.data

    def save_to_json(self, filename):
        if self.is_narrow:
            # in this case we have only one ball in are model data
            data = NarrowModelData(
                radius=self.data.radius,
                ball_number=self.ball_index_to_follow + 1,
                first_quarter_probability=self.data.get_first_quarter_probability(0),
                x_velocity_distribution=self.data.get_vx(0, is_distribution=True),
            )
        else:
            data = self.data
        with open(filename, "wt") as fd:
            json.dump(data.to_dict(), fd, indent=4)

    def _forward(self, dt):
        time_to_next_store = self.dtstore - self.time % self.dtstore
        if dt >= time_to_next_store:
            self._move(time_to_next_store)
            self._store_state()
            dt -= time_to_next_store

        self._move(dt)

    def _move(self, dt):
        self.time += dt
        for ball in self.balls:
            ball.move(dt)

    def _find_next_collision(self):
        min_collision_time = float("inf")
        collsion_event = None
        for ball_1, ball_2 in itertools.combinations(self.balls, 2):
            dtcoll = collisions.find_dtcoll(ball_1, ball_2)
            if dtcoll < min_collision_time:
                min_collision_time = dtcoll
                collsion_event = BallsCollision(ball_1, ball_2, dtcoll)
        for ball in self.balls:
            wall, dtwall = collisions.find_wall_and_dt_wall(ball)
            if dtwall < min_collision_time:
                min_collision_time = dtwall
                collsion_event = WallCollision(ball, wall, dtwall)

        return collsion_event

    def _store_state(self):
        if self.is_narrow:
            self.data.add_state(
                ModelState(
                    balls=[
                        Ball(
                            x=self.balls[self.ball_index_to_follow].x,
                            y=self.balls[self.ball_index_to_follow].y,
                            vx=self.balls[self.ball_index_to_follow].vx,
                            vy=self.balls[self.ball_index_to_follow].vy,
                            radius=self.balls[self.ball_index_to_follow].radius,
                        )
                    ],
                    time=self.time,
                )
            )
        else:
            self.data.add_state(
                ModelState(
                    balls=[
                        Ball(
                            x=ball.x,
                            y=ball.y,
                            vx=ball.vx,
                            vy=ball.vy,
                            radius=ball.radius,
                            name=ball.name,
                        )
                        for ball in self.balls
                    ],
                    time=self.time,
                )
            )

    def __str__(self):
        return f"Model(balls={self.balls}, size={self.size}, dtstore={self.dtstore})"
