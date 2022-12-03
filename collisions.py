import numpy as np
from numpy.linalg import norm


def find_dtwall(ball):
    x, y, vx, vy, r = ball.x, ball.y, ball.vx, ball.vy, ball.radius
    if vx > 0:
        dtwall_x = (1 - r - x) / vx
    elif vx < 0:
        dtwall_x = (x - r) / abs(vx)
    else:
        dtwall_x = 10 ** 10 # Practically infinity

    if vy > 0:
        dtwall_y = (1 - r - y) / vy
    elif vx < 0:
        dtwall_y = (y - r) / abs(vy)
    else:
        dtwall_y = 10 ** 10 # Practically infinity

    return min(dtwall_x, dtwall_y)


def find_dtcoll(ball_1, ball_2):
    r = ball_1.radius

    delta_x = ball_2.x - ball_1.x
    delta_y = ball_2.y - ball_1.y
    delta_l = np.array([delta_x, delta_y])

    delta_vx = ball_2.vx - ball_1.vx
    delta_vy = ball_2.vy - ball_1.vy
    delta_v = np.array(delta_vx, delta_vy)

    s = np.dot(delta_v, delta_l)

    det = s ** 2 - (norm(delta_v) ** 2) * (norm(delta_l) ** 2 - 4 * r ** 2)

    if det > 0 and s< 0:
        dtcoll = - (s + np.sqrt(det)) / (norm(delta_v) ** 2)
    else:
        dtcoll = 10 ** 10 # Practically infinity

    return dtcoll


def make_wall_collision(ball):
    return 0.0, 0.0


def make_balls_collision(ball_1, ball_2):
    delta_x = ball_2.x - ball_1.x
    delta_y = ball_2.y - ball_1.y
    delta_l = np.array([delta_x, delta_y])

    delta_vx = ball_2.vx - ball_1.vx
    delta_vy = ball_2.vy - ball_1.vy
    delta_v = np.array(delta_vx, delta_vy)

    e_x = delta_x / norm(delta_l)
    e_y = delta_y / norm(delta_l)
    e_vec = np.array([e_y, e_y])

    s_v = np.dot(delta_v, e_vec)

    # Modify velocities:
    ball_1.vx = ball_1.vx + e_x * s_v
    ball_1.vy = ball_1.vy + e_y * s_v
    ball_2.vx = ball_2.vx - e_x * s_v
    ball_2.vy = ball_2.vy - e_y * s_v
