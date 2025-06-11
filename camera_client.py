from multiprocessing.connection import Client
from shared_classes import Ball, Message
import cv2, sys, math, numpy as np
import threading, time

DIST_COEF = 10
F_PX = 500
REAL_CM = 6.5
LOWER = np.array([20, 40, 80])
UPPER = np.array([50, 255, 255])

idx = 0
pending_balls = None
pending_lock  = threading.Lock()
latest_labels = {}
labels_lock = threading.Lock()
stop_event = threading.Event()

def new_idx():
    global idx
    idx += 1
    return idx

def network_thread(conn):
    global pending_balls, latest_labels
    while not stop_event.is_set():
        time.sleep(0.01)
        with pending_lock:
            balls_to_send = pending_balls
        if balls_to_send is None:
            continue

        try:
            conn.send(Message(balls_to_send))
            labels = conn.recv()
        except (EOFError, OSError, BrokenPipeError):
            print("[WORKER] utracono połączenie, zatrzymuję sieć")
            stop_event.set()
            break

        with labels_lock:
            latest_labels = labels

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


def find_balls_on_frame(frame, cr, sr, x_cam, y_cam, z_cam):
    balls = []

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = merge([cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 800])
    # cv2.imshow("Mask", cv2.resize(mask, None, fx=1.5, fy=1.5))
    
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

        rx = X_cam * cr + Z_cam * sr + x_cam
        rz = -X_cam * sr + Z_cam * cr + z_cam
        ry = Y_cam + y_cam
        balls.append(Ball(cx, cy, w, h, rx, ry, rz))
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


def receive_labels(conn):
    try:
        labels = conn.recv()
    except EOFError:
        print("[WORKER] Server closed connection, exiting...")
        sys.exit(0)
    print(labels)
    return labels


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
        if backend is not None:
            cap = cv2.VideoCapture(idx, backend)
        else:
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ret, _ = cap.read()
        cap.release()
        if ret:
            return idx


def main(host, port, authkey, x_cam, y_cam, z_cam, r):
    global pending_balls
    cr, sr = math.cos(r), math.sin(r)

    cam_idx = get_cam_idx()
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print('Camera has not been found')
        return

    balls = []

    conn = Client((host, port), authkey=authkey)
    net_t = threading.Thread(target=network_thread, args=(conn,), daemon=True)
    net_t.start()
    print('Established connection')

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        new_balls = find_balls_on_frame(frame, cr, sr, x_cam, y_cam, z_cam)
        balls = find_common_balls(balls, new_balls)
        with pending_lock:
            pending_balls = list(balls)

        with labels_lock:
            labels = latest_labels.copy()

        draw_rects(frame, balls, labels)
        if cv2.waitKey(1) & 0xFF == 27:
            stop_event.set()
            net_t.join()
            conn.close()
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
