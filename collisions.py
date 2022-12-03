def find_dtwall(ball):
    x, y, vx, vy, r = ball.x, ball.y, ball.vx, ball.vy, ball.radius
    if vx > 0:
        dtwall_x = (1 - r - x) / vx
    else:
        dtwall_x = (x - r) / vx

    if vy > 0:
        dtwall_y = (1 - r - y) / vy
    else:
        dtwall_y = (y - r) / vy

    return min(dtwall_x, dtwall_y)


def find_dtcol(ball_1, ball_2):
    # Find the distance to the collision
    # with the infrared sensor
    # and return the distance
    return 0.0


def make_collision(ball_1, ball_2):
    # Make the collision
    # between the two balls
    # and return the new velocity
    # of the two balls
    return 0.0, 0.0
