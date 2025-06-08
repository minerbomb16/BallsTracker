import cv2, sys, math, numpy as np, time
from multiprocessing.connection import Client
from shared_classes import Message

MAX_MISS = 10
DIST_COEF = 10
F_PX = 500
REAL_CM = 6.0
LOWER = np.array([20, 40, 80])
UPPER = np.array([50, 255, 255])

R_MM = 33.5
w, h = 1920, 1080
idx = 0


def new_idx():
    global idx
    idx += 1
    return idx


def merge(rects):
    merged = []
    used = [False] * len(rects)
    for i, (x, y, w, h) in enumerate(rects):
        if used[i]:
            continue
        rx, ry, rw, rh = x, y, w, h
        for j in range(i + 1, len(rects)):
            if used[j]:
                continue
            x2, y2, w2, h2 = rects[j]
            if not (rx + rw < x2 or x2 + w2 < rx or ry + rh < y2 or y2 + h2 < ry):
                rx = min(rx, x2)
                ry = min(ry, y2)
                rw = max(rx + rw, x2 + w2) - rx
                rh = max(ry + rh, y2 + h2) - ry
                used[j] = True
        used[i] = True
        merged.append((rx, ry, rw, rh))
    return merged


class Ball:
    def __init__(self, cx, cy, width, height, x, y, z):
        self.cx = cx
        self.cy = cy
        self.w = width
        self.h = height
        self.miss = 0
        self.label = '?'
        self.x = x
        self.y = y
        self.z = z


def find_balls_on_frame(frame, cr, sr):
    balls = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = merge([cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 800])
    for x, y, w, h in rects:
        cx = x + w // 2
        cy = y + h // 2

        h_img, w_img = frame.shape[:2]
        cx0 = w_img // 2
        cy0 = h_img // 2
        scale = REAL_CM / max(w, h)

        Z_cam = scale * F_PX
        X_cam = (cx0 - cx) * scale
        Y_cam = (cy0 - cy) * scale

        x = X_cam * cr + Z_cam * sr + x_cam
        y = Y_cam + y_cam
        z = -X_cam * sr + Z_cam * cr + z_cam
        balls.append(Ball(cx, cy, w, h, x, y, z))
    return balls


def find_common_balls(prev_balls, new_balls):
    res = []
    used = set()
    for n_ball in new_balls:
        thr = max(n_ball.w, n_ball.h) * DIST_COEF
        closest_dist = thr
        closest_ball = None
        for p_ball in prev_balls:
            if p_ball.index in used:
                continue
            d = math.hypot(n_ball.cx - p_ball.cx, n_ball.cy - p_ball.cy)
            if d < closest_dist:
                closest_dist = d
                closest_ball = p_ball
        if closest_ball:
            used.add(closest_ball.index)
            n_ball.index = closest_ball.index
        else:
            n_ball.index = new_idx()
        res.append(n_ball)
    return res


def send_balls(balls, conn):
    msg = Message(balls)
    inds = [b.index for b in balls]
    print(inds)
    conn.send(msg)


# returns a dictionary with ball idx as keys and labels as values
def receive_labels(conn):
    labels = conn.recv()
    print(labels)
    return labels


def process_frame(frame, balls):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = merge([cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 800])
    for x, y, w, h in rects:
        cx = x + w // 2
        cy = y + h // 2
        thr = max(w, h) * DIST_COEF
        best = None
        best_d = thr
        for b in balls:
            d = math.hypot(b.cx - cx, b.cy - cy)
            if d < best_d:
                best = b
                best_d = d
        if best:
            best.cx = cx
            best.cy = cy
            best.rect = (x, y, w, h)
            best.miss = 0
        else:
            balls.append(Ball(cx, cy, (x, y, w, h)))
    return balls


def draw_rects(frame, balls, labels):
    for b in balls:
        label = labels[b.index] if b.index in labels else '?'
        red = (0, 0, 255)
        x1 = b.cx - b.w // 2
        y1 = b.cy - b.h // 2
        x2 = b.cx + b.w // 2
        y2 = b.cy + b.h // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), red, 2)
        lines = [label, f"X:{b.x:.1f}", f"Y:{b.y:.1f}", f"Z:{b.z:.1f}"]
        for i, line in enumerate(lines):
            cords = x2 + 10, y1 + 20 + i * 25
            cv2.putText(frame, line, cords, cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 1)
    cv2.imshow("Balls", cv2.resize(frame, None, fx=1.5, fy=1.5))


def get_cam_idx(max_index=10, backend=None):
    for idx in range(max_index):
        # try to open the capture device
        if backend is not None:
            cap = cv2.VideoCapture(idx, backend)
        else:
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        # optional: try to grab a frame to be extra sure
        ret, _ = cap.read()
        cap.release()
        if ret:
            return idx


def main(host, port, authkey, x_cam, y_cam, z_cam, r):
    cr, sr = math.cos(r), math.sin(r)

    idx = get_cam_idx()
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print('Camera has not been found')
        return

    balls = []

    conn = Client((host, port), authkey=authkey)
    print('Estabilished connection')

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        new_balls = find_balls_on_frame(frame, cr, sr)
        balls = find_common_balls(balls, new_balls)
        send_balls(balls, conn)
        labels = receive_labels(conn)
        draw_rects(frame, balls, labels)
        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("python worker.py host:port authkey x y z r")
        exit(-1)
    host, port_str = sys.argv[1].split(':')
    port = int(port_str)
    authkey = sys.argv[2].encode()
    x_cam, y_cam, z_cam = map(float, sys.argv[3:6])
    r = math.radians(float(sys.argv[6]))
    main(host, port, authkey, x_cam, y_cam, z_cam, r)
