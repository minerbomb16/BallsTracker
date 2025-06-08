from multiprocessing.connection import Listener
from shared_classes import Message
import math, sys, socket
import time
import threading

MAX_MISS = 20
START_L = ord('A')
END_L = ord('Z')
curr_label = ord('A')
labels = dict()
labels_lock = threading.Lock()


class Label:
    def __init__(self, x, y, z, client):
        self.x = x
        self.y = y
        self.z = z
        self.clients = set()
        self.clients.add(client)

    def update_pos(self, ball):
        self.x = ball.x
        self.y = ball.y
        self.z = ball.z

    def add_client(self, client):
        if client not in self.clients:
            self.clients.add(client)

    def remove_client(self, client):
        if client in self.clients:
            self.clients.remove(client)

    def client_count(self):
        return len(self.clients)


def label_balls(balls, client: str):
    labels_lock.acquire()
    ind_labs = dict()

    found_labels: dict[str, float] = dict()
    for ball in balls:
        closest_dist = math.inf
        closest_label = None
        for lab in labels.keys():
            label = labels[lab]
            d = dist3(ball, label)
            if d < closest_dist:
                if lab not in found_labels or found_labels[lab] > d:
                    if lab in found_labels:
                        ind_labs = {i : l for i, l in ind_labs.items() if l != lab}
                    closest_label = lab
                    closest_dist = d
                    found_labels[lab] = d
                        
        if closest_label:
            ind_labs[ball.index] = closest_label
            label = labels[closest_label]
            label.update_pos(ball)
        else:
            new_l = next_free()
            labels[new_l] = Label(ball.x, ball.y, ball.z, client)
            ind_labs[ball.index] = new_l
            found_labels[new_l] = 0

    to_delete = set()
    for lab in labels.keys():
        if lab in found_labels:
            continue
        else:
            label = labels[lab]
            label.remove_client(client)
            if label.client_count() == 0:
                to_delete.add(lab)
    for lab in to_delete:
        labels.pop(lab)

    for idx, lab in ind_labs.items():
        ball = next(b for b in balls if b.index == idx)
        label = labels[lab]
        label.update_pos(ball)

    print(labels.keys())
    for lab in labels.keys():
        label = labels[lab]
        print(f'{lab}: {label.client_count()}')
    labels_lock.release()
    return ind_labs


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


def dist3(b, label):
    return math.sqrt((b.x - label.x)**2 + (b.y - label.y)**2 + (b.z - label.z)**2)


def listen_on_connections(host, port, key, conn_threads, stop, tracked):
    conn_id = 0
    listener = Listener((host, port), authkey=key)
    # listener._listener._socket.settimeout(1.0)
    print(f"[SERVER] listening on {host}:{port}")
    while not stop:
        conn = listener.accept()
        print('Estabilished connection')
        client = f'client {conn_id}'
        conn_id += 1
        new_t = threading.Thread(target=serve, args=(conn, client, stop))
        new_t.start()
        conn_threads.append(new_t)
    listener.close()


def serve(conn, client, stop):
    print(f'Spawned thread for {client}')
    labels = dict()
    while not stop:
        message = conn.recv()
        labels = label_balls(message.balls, client)
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
