import cv2, sys, math, numpy as np
from multiprocessing.connection import Client

SEND_EVERY = 3
MAX_MISS = 10
DIST_COEF = 3.5
F_PX = 500
REAL_CM = 6.0
LOWER = np.array([20, 40, 80])
UPPER = np.array([50, 255, 255])

class Ball:
    def __init__(s, idx, cx, cy, rect):
        s.idx = idx
        s.cx = cx
        s.cy = cy
        s.rect = rect
        s.miss = 0
        s.label = '?'
        s.x = s.y = s.z = 0.0

def merge(rs):
    out, used = [], [False]*len(rs)
    for i, (x, y, w, h) in enumerate(rs):
        if used[i]: continue
        rx, ry, rw, rh = x, y, w, h
        for j in range(i + 1, len(rs)):
            if used[j]: continue
            x2, y2, w2, h2 = rs[j]
            if not (rx + rw < x2 or x2 + w2 < rx or ry + rh < y2 or y2 + h2 < ry):
                rx = min(rx, x2)
                ry = min(ry,y2)
                rw = max(rx + rw, x2 + w2) - rx
                rh = max(ry + rh, y2 + h2) - ry
                used[j] = True
        used[i] = True
        out.append((rx, ry, rw, rh))
    return out

def process_frame(frame, balls, idx_counter):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = merge([cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 800])

    for x, y, w, h in rects:
        cx = x + w // 2
        cy = y + h // 2
        thr = max(w, h) * DIST_COEF
        best, best_d = None, thr
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
            balls.append(Ball(idx_counter, cx, cy, (x, y, w, h)))
            idx_counter += 1

    return [b for b in balls if b.miss <= MAX_MISS], idx_counter

def main():
    if len(sys.argv)<7:
        print("python worker.py host:port authkey x y z r"); return
    host, port = sys.argv[1].split(':')
    port = int(port)
    authkey = sys.argv[2].encode()
    x_cam, y_cam, z_cam = map(float, sys.argv[3:6])
    r = math.radians(float(sys.argv[6]))
    cr, sr = math.cos(r), math.sin(r)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened(): print("camera?"); return
    conn = Client((host, port), authkey=authkey)
    balls = []
    frame_idx = 0
    idx_counter = 0

    try:
        while True:
            ok, frame = cam.read()
            if not ok: break

            for b in balls: b.miss += 1

            frame = cv2.flip(frame, 1)
            balls, idx_counter = process_frame(frame, balls, idx_counter)
            h_img, w_img = frame.shape[:2]
            cx0 = w_img // 2
            cy0 = h_img // 2

            for b in balls:
                x, y, w, h = b.rect
                scale = REAL_CM / max(w, h)
                Z = scale * F_PX
                X = (cx0 - b.cx) * scale
                Y = (cy0 - b.cy) * scale
                b.x = X * cr + Z * sr + x_cam
                b.y =  Y + y_cam
                b.z = -X * sr + Z * cr + z_cam

            frame_idx += 1
            if frame_idx % SEND_EVERY == 0:
                pkt = [b for b in balls]
                try:
                    conn.send(pkt)
                    mapping = conn.recv()
                    for idx, lab in mapping:
                        for b in balls:
                            if b.idx == idx: b.label = lab
                except Exception as e:
                    print("Send error:", e)

            for b in balls:
                x, y, w, h = b.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                for i, line in enumerate([f"Ball {b.label}",
                                          f"X:{b.x:.1f}",
                                          f"Y:{b.y:.1f}",
                                          f"Z:{b.z:.1f}"]):
                    cv2.putText(frame, line, (x + w + 10, y + 20 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            cv2.imshow("balls", cv2.resize(frame, None, fx=1.5, fy=1.5))
            if cv2.waitKey(1) & 0xFF == 27: break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        conn.close()

if __name__=="__main__":
    main()
