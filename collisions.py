import numpy as np

def find_dtwall(ball):
    x, y, vx, vy, r = ball.x, ball.y, ball.vx, ball.vy, ball.radius
    if vx > 0:
        dtwall_x = (1 - r - x) / vx
    elif vx < 0
        dtwall_x = (x - r) / abs(vx)
    else:
        dtwall_x = 10 ** 10 # Practically infinity

    if vy > 0:
        dtwall_y = (1 - r - y) / vy
    elif vx < 0
        dtwall_y = (y - r) / abs(vy)
    else:
        dtwall_y = 10 ** 10 # Practically infinity

    return min(dtwall_x, dtwall_y)


def find_dtcoll(ball_1, ball_2):
    delta_x = ball_2.x - ball_1.x
    delta_y = ball_2.y - ball_1.y
    delta_l = np.array([delta_x, delta_y])

    delta_vx = ball_2.vx - ball_1.vx
    delta_vy = ball_2.vy - ball_1.vy
    delta_v = np.array(delta_vx, delta_vy)

    s = np.dot(delta_v, delta_l)

def make_wall_collision(ball):
    return 0.0, 0.0

def make_balls_collision(ball_1, ball_2):
    return 0.0, 0.0
