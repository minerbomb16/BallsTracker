import math, sys, socket, threading, time
from multiprocessing.connection import Listener

POS_THR = 10.0 * 2
MAX_MISS = 20
START_L = ord('A')
END_L = ord('Z')

class Ball:
    def __init__(self, label, x, y, z, idx = None):
        self.label = label
        self.x, self.y, self.z = x, y, z
        self.miss = 0
        self.idx = idx

def next_free(tracked):
    used = {b.label for b in tracked}
    for c in range(START_L, END_L + 1):
        if chr(c) not in used:
            return chr(c)
    return '?'

def dist3(b1, b2):
    return math.sqrt((b1.x - b2.x) ** 2 + (b1.y - b2.y) ** 2 + (b1.z - b2.z) ** 2)

def serve(conn, label, stop_flag, tracked):
    try:
        while not stop_flag.is_set():
            try:
                msg = conn.recv()
            except EOFError:
                break

            result = process(msg, tracked)
            conn.send(result)
    finally:
        conn.close()
        print(f"[SERVER] {label} disconnected")

def process(r_balls, tracked):
    for b in tracked:
        b.miss += 1

    labels_out = []
    for r in r_balls:
        best, best_d = None, POS_THR
        for t in tracked:
            d = dist3(r, t)
            if d < best_d:
                best, best_d = t, d
        if best:
            best.x, best.y, best.z, best.miss = r.x, r.y, r.z, 0
            best.idx = r.idx  
            lab = best.label
        else:
            lab = next_free(tracked)
            tracked.append(Ball(lab, r.x, r.y, r.z, idx=r.idx))
        labels_out.append((r.idx, lab))

    tracked[:] = [b for b in tracked if b.miss <= MAX_MISS]
    return labels_out

def listen(host, port, auth, stop_flag, tracked):
    listener = Listener((host, port), authkey=auth)
    listener._listener._socket.settimeout(1.0)
    print(f"[SERVER] listening on {host}:{port}")

    conn_id = 0
    threads = []
    try:
        while not stop_flag.is_set():
            try:
                conn = listener.accept()
            except socket.timeout:
                continue
            label = f"client{conn_id}"
            conn_id += 1
            t = threading.Thread(target=serve, args=(conn, label, stop_flag, tracked), daemon=True)
            t.start()
            threads.append(t)
    finally:
        listener.close()
        for t in threads: t.join()

def main():
    if len(sys.argv) < 3:
        print("python server.py host:port authkey"); return
    host, port = sys.argv[1].split(':'); port = int(port)
    authkey = sys.argv[2].encode()

    stop_flag = threading.Event()
    tracked = []

    try:
        listen(host, port, authkey, stop_flag, tracked)
    except KeyboardInterrupt:
        print("\n[SERVER] stopping …")
        stop_flag.set()

if __name__ == "__main__":
    main()
