import math, sys, socket
from multiprocessing.connection import Listener
import time
import threading

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


def dist3(b1, b2):
    return math.sqrt((b1.x - b2.x) ** 2 + (b1.y - b2.y) ** 2 + (b1.z - b2.z) ** 2)


def listen_on_connections(host, port, key, conn_threads, stop, tracked):
    conn_id = 0
    listener = Listener((host, port), authkey=key)
    # listener._listener._socket.settimeout(1.0)
    print(f"[SERVER] listening on {host}:{port}")
    while not stop:
        conn = listener.accept()
        label = f'client {conn_id}'
        conn_id += 1
        new_t = threading.Thread(target=serve, args=(conn, label, stop, tracked))
        conn_threads.append(new_t)
    listener.close()


def serve(conn, label, stop, tracked):
    while not stop:
        message = conn.receive()
        print('Received message from', label)
        result = process(message)
        conn.send(result)
    conn.close()


def listen_for_messages(conn, label, recv_que, recv_lock, stop):
    while not stop:
        message = conn.recv()
        print('Received messsage from', label)
        recv_lock.acquire()
        recv_que.append((message, label))
        recv_lock.release()


def send_messages(clients, send_que, send_lock, stop):
    while not stop:
        if len(send_que) == 0:
            time.sleep(0.01)
            continue
        send_lock.acquire()
        msg, client = send_que.pop()
        send_lock.release()
        clients[client].send(msg)
        print(f'Sent message to {client}')


def process(msg, tracked):
    for b in tracked:
        b.miss += 1

    labels_out = []
    for r_ball in msg.balls:
        best = None
        best_d = POS_THR
        for i, t_ball in enumerate(tracked):
            d = dist3(r_ball, t_ball)
            if d < best_d:
                best = i
                best_d = d
        if best:
            label = tracked[best].label
            tracked[best] = r_ball
            labels_out.append((r_ball.index, label))
        else:
            # new ball
            r_ball.label = next_free(tracked)
            tracked.append(r_ball)
            labels_out.append((r_ball.index, r_ball.label))
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
