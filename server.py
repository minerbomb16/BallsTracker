from multiprocessing.connection import Listener
from shared_classes import Message
import math, sys, socket
import time
import threading

POS_THR = 10.0
MAX_MISS = 20
START_L = ord('A')
END_L = ord('Z')
curr_label = ord('A')
labels = []


def label_balls(balls):
    pass


class Ball:
    def __init__(self, label, x, y, z):
        self.label = label
        self.x, self.y, self.z = x, y, z
        self.miss = 0


def next_free():
    global curr_label, prefix
    c = chr(curr_label)
    curr_label += 1
    if curr_label > ord('Z'):
        curr_label = ord('A')
    return c


def dist3(b1, b2):
    return math.sqrt((b1.x - b2.x)**2 + (b1.y - b2.y)**2 + (b1.z - b2.z)**2)


def listen_on_connections(host, port, key, conn_threads, stop, tracked):
    conn_id = 0
    listener = Listener((host, port), authkey=key)
    # listener._listener._socket.settimeout(1.0)
    print(f"[SERVER] listening on {host}:{port}")
    while not stop:
        conn = listener.accept()
        print('Estabilished connection')
        label = f'client {conn_id}'
        conn_id += 1
        new_t = threading.Thread(target=serve, args=(conn, label, stop, tracked))
        new_t.start()
        conn_threads.append(new_t)
    listener.close()


def serve(conn, label, stop, tracked):
    print(f'Spawned thread for {label}')
    labels = dict()
    while not stop:
        message = conn.recv()
        labels = process(message, tracked, labels)
        conn.send(labels)
    conn.close()


def process(msg, tracked, labels):
    labels_out = {}

    for r_ball in msg.balls:
        best_i  = None
        best_d  = POS_THR
        for i, t_ball in enumerate(tracked):
            d = dist3(r_ball, t_ball)
            if d < best_d:
                best_i, best_d = i, d

        if best_i is not None:
            t = tracked[best_i]
            t.x, t.y, t.z = r_ball.x, r_ball.y, r_ball.z
            label = t.label
        else:
            label = next_free()
            tracked.append(Ball(label, r_ball.x, r_ball.y, r_ball.z))

        labels_out[r_ball.index] = label
        labels[r_ball.index] = label

    return labels_out


def main():
    if len(sys.argv) < 3:
        print("python server.py host:port authkey")
        return

    host, port = sys.argv[1].split(':')
    port = int(port)
    authkey = sys.argv[2].encode()

    try:
        stop = False
        conn_threads = []
        tracked = []
        listener_t = threading.Thread(target=listen_on_connections, args=(host, port, authkey, conn_threads, stop, tracked))
        listener_t.start()

        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop = True
        listener_t.join()
        for t in conn_threads:
            t.join()


if __name__ == "__main__":
    main()
