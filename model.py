import dataclasses
import itertools
import collisions
import logging
from typing import List
import copy

logging.basicConfig(level=logging.DEBUG)


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
    def vabs(self):
        return (self.vx ** 2 + self.vy ** 2) ** 0.5

    def __str__(self):
        return f"Ball(x={self.x}, y={self.y}, vx={self.vx}, vy={self.vy}, radius={self.radius})"


@dataclasses.dataclass
class BallsCollision:
    ball_1: Ball
    ball_2: Ball
    dt: float

    def __str__(self):
        return (
            f"BallsCollision(ball_1={self.ball_1}, ball_2={self.ball_2}, dt={self.dt})"
        )


@dataclasses.dataclass
class WallCollision:
    ball: Ball
    dt: float

    def __str__(self):
        return f"WallCollision(ball={self.ball}, dt={self.dt})"


@dataclasses.dataclass
class ModelState:
    balls: list
    time: float


@dataclasses.dataclass
class ModelData:
    model_states: List[ModelState]

    def add_state(self, state: ModelState):
        self.model_states.append(state)

    @property
    def times(self):
        return [state.time for state in self.model_states]

    def get_x(self, ball_number):
        return [state.balls[ball_number].x for state in self.model_states]

    def get_y(self, ball_number):
        return [state.balls[ball_number].y for state in self.model_states]

    def get_vabs(self, ball_number):
        return [state.balls[ball_number].vabs for state in self.model_states]

    def get_vx(self, ball_number):
        return [state.balls[ball_number].vx for state in self.model_states]

    def get_vy(self, ball_number):
        return [state.balls[ball_number].vy for state in self.model_states]


class Model:
    def __init__(self, balls, size, dtstore):
        self.balls = balls
        self.size = size
        self.dtstore = dtstore
        self.time = 0
        self.data = ModelData(model_states=[])

    def run(self, steps=100):
        step = 0
        while step <= steps:
            step += 1
            collision = self._find_next_collision()
            self._forward(collision.dt)
            if isinstance(collision, BallsCollision):
                collisions.make_balls_collision(collision.ball_1, collision.ball_2)
            elif isinstance(collision, WallCollision):
                collisions.make_wall_collision(collision.ball)

            if step % 1 == 0:
                logging.debug(f"step={step}, {collision}")

        return self.data

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
        collision_events = []
        for ball_1, ball_2 in itertools.combinations(self.balls, 2):
            collision_events.append(
                BallsCollision(
                    ball_1=ball_1,
                    ball_2=ball_2,
                    dt=collisions.find_dtcoll(ball_1, ball_2),
                )
            )
        for ball in self.balls:
            collision_events.append(
                WallCollision(ball=ball, dt=collisions.find_dtwall(ball),)
            )
        return min(collision_events, key=lambda x: x.dt)

    def _store_state(self):
        logging.debug(f"store state at time={self.time}")
        self.data.add_state(ModelState(balls=copy.deepcopy(self.balls), time=self.time))

    def __str__(self):
        return f"Model(balls={self.balls}, size={self.size}, dtstore={self.dtstore})"
