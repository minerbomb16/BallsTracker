import math, sys, socket
from multiprocessing.connection import Listener

POS_THR = 10.0
MAX_MISS = 20
START_L = ord('A')
END_L = ord('Z')

class Ball:
    def __init__(self, label, x, y, z):
        self.label = label
        self.x, self.y, self.z = x, y, z
        self.miss = 0

def next_free(tracked):
    used = {b.label for b in tracked}
    for c in range(START_L, END_L + 1):
        if chr(c) not in used:
            return chr(c)
    return '?'

def dist3(b, p):
    return math.sqrt((b.x - p[0]) ** 2 + (b.y - p[1]) ** 2 + (b.z - p[2]) ** 2)

def main():
    if len(sys.argv) < 3:
        print("python server.py host:port authkey"); return
    host, port = sys.argv[1].split(':')
    port = int(port)
    authkey = sys.argv[2].encode()
    listener = Listener((host, port), authkey=authkey)
    listener._listener._socket.settimeout(1.0)
    print(f"[SERVER] listening on {host}:{port}")

    tracked = []

    try:
        while True:
            try:
                conn = listener.accept()
            except socket.timeout:
                continue
            points = conn.recv()
            if not isinstance(points, list):
                conn.close(); continue

            for b in tracked: b.miss += 1

            labels_out = []
            for p in points:
                best = None
                best_d = POS_THR
                for b in tracked:
                    d = dist3(b, p)
                    if d < best_d:
                        best = b
                        best_d = d
                if best:
                    best.x, best.y, best.z, best.miss = *p, 0
                    labels_out.append(best.label)
                else:
                    lab = next_free(tracked)
                    tracked.append(Ball(lab, *p))
                    labels_out.append(lab)

            tracked = [b for b in tracked if b.miss <= MAX_MISS]
            conn.send(labels_out)
            conn.close()
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        listener.close()

if __name__ == "__main__":
    main()
