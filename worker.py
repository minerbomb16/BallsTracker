import cv2, sys, math, numpy as np
from multiprocessing.connection import Client

SEND_EVERY = 2
MAX_MISS = 10
DIST_COEF = 1.2
F_PX = 500
REAL_CM = 6.0
LOWER = np.array([20,40,80])
UPPER = np.array([50,255,255])

def merge(rects):
    merged, used = [], [False]*len(rects)
    for i, (x, y, w, h) in enumerate(rects):
        if used[i]: continue
        rx, ry, rw, rh = x, y, w, h
        for j in range(i + 1, len(rects)):
            if used[j]: continue
            x2, y2, w2, h2 = rects[j]
            if not (rx + rw < x2 or x2 + w2 < rx or ry + rh < y2 or y2 + h2 < ry):
                rx = min(rx, x2)
                ry = min(ry, y2)
                rw = max(rx + rw, x2 + w2) - rx
                rh = max(ry + rh, y2 + h2) - ry
                used[j] = True
        used[i]=True
        merged.append((rx, ry, rw, rh))
    return merged

class Ball:
    def __init__(s, cx, cy, rect):
        s.cx = cx
        s.cy = cy
        s.rect = rect
        s.miss = 0
        s.label = '?'
        s.X = s.Y = s.Z = 0.0

def main():
    if len(sys.argv)<7:
        print("python worker.py host:port authkey x y z r"); return
    host, port = sys.argv[1].split(':')
    port=int(port)
    AUTHKEY = sys.argv[2].encode()
    x_cam, y_cam, z_cam = map(float, sys.argv[3:6])
    r = math.radians(float(sys.argv[6]))
    cr, sr = math.cos(r), math.sin(r)

    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): print("kamera?"); return

    balls=[]
    frame_idx=0

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER,UPPER)
        cnts, _ =cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = merge([cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 800])

        for b in balls: b.miss += 1

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

        balls=[b for b in balls if b.miss <= MAX_MISS]

        h_img, w_img = frame.shape[:2]
        cx0 = w_img // 2
        cy0 = h_img // 2
        for b in balls:
            x, y, w, h = b.rect
            scale = REAL_CM / max(w, h)

            Z_cam = scale * F_PX
            X_cam = (cx0 - b.cx) * scale
            Y_cam = (cy0 - b.cy) * scale

            b.X = X_cam * cr + Z_cam * sr + x_cam
            b.Y = -X_cam * sr + Z_cam * cr + z_cam
            b.Z = Y_cam + y_cam

        frame_idx += 1
        if frame_idx % SEND_EVERY == 0:
            try:
                conn = Client((host, port), authkey = AUTHKEY)
                conn.send([(b.X, b.Y, b.Z) for b in balls])
                labels = conn.recv()
                conn.close()
                if len(labels) == len(balls):
                    for b, l in zip(balls, labels): b.label = l
            except Exception:
                pass

        for b in balls:
            x, y, w, h = b.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            for i, line in enumerate([f"Ball {b.label}",
                                      f"X:{b.X:.1f}",
                                      f"Y:{b.Y:.1f}",
                                      f"Z:{b.Z:.1f}"]):
                cv2.putText(frame, line, (x + w + 10, y + 20 + i *25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.imshow("Balls", cv2.resize(frame, None, fx = 1.5, fy = 1.5))
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
