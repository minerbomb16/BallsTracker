class Ball:
    def __init__(self, cx, cy, width, height, x, y, z, index=None):
        self.cx = cx
        self.cy = cy
        self.w = width
        self.h = height
        self.label = '?'
        self.x = x
        self.y = y
        self.z = z
        self.index = index

class Message():
    def __init__(self, balls):
        self.balls = balls
