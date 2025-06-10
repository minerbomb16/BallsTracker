from multiprocessing.connection import Listener
from shared_classes import Ball, Message
import math, sys, socket, time, threading

MAX_LABEL_DIST = 15.0 
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
    ind_labs: dict[int, str] = {}

    with labels_lock:
        matched: set[str] = set()

        for ball in balls:
            closest_label = None
            closest_dist = MAX_LABEL_DIST

            for lab, lab_obj in labels.items():
                if lab in matched:
                    continue
                d = dist3(ball, lab_obj)
                if d < closest_dist:
                    closest_dist = d
                    closest_label = lab

            if closest_label is not None:
                ind_labs[ball.index] = closest_label
                label_obj = labels[closest_label]
                label_obj.update_pos(ball)
                label_obj.add_client(client)
                matched.add(closest_label)
            else:
                new_lab = next_free()
                labels[new_lab] = Label(ball.x, ball.y, ball.z, client)
                ind_labs[ball.index] = new_lab
                matched.add(new_lab)

        to_delete = []
        for lab, lab_obj in labels.items():
            if lab not in matched:
                lab_obj.remove_client(client)
                if lab_obj.client_count() == 0:
                    to_delete.append(lab)

        for lab in to_delete:
            labels.pop(lab)

    return ind_labs


def next_free():
    global curr_label
    c = chr(curr_label)
    curr_label += 1
    if curr_label > END_L:
        curr_label = START_L
    return c


def dist3(b, label):
    return math.sqrt((b.x - label.x)**2 + (b.y - label.y)**2 + (b.z - label.z)**2)


def listen_on_connections(host, port, authkey, conn_threads, stop_event):
    listener = Listener((host, port), authkey=authkey)
    listener._listener._socket.settimeout(1.0)
    print(f"[SERVER] listening on {host}:{port}")

    conn_id = 0
    try:
        while not stop_event.is_set():
            try:
                conn = listener.accept()
            except socket.timeout:
                continue
            print(f"[SERVER] connection #{conn_id} established")
            client_name = f"client {conn_id}"
            t = threading.Thread(target=serve, args=(conn, client_name, stop_event))
            t.start()
            conn_threads.append(t)
            conn_id += 1
    finally:
        listener.close()
        print("[SERVER] listener closed")


def serve(conn, client, stop_event):
    print(f"[SERVER] started thread for {client}")
    try:
        while not stop_event.is_set():
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                break

            labels_map = label_balls(msg.balls, client)
            conn.send(labels_map)
    finally:
        with labels_lock:
            to_remove = []
            for lab, lab_obj in labels.items():
                lab_obj.remove_client(client)
                if lab_obj.client_count() == 0:
                    to_remove.append(lab)
            for lab in to_remove:
                labels.pop(lab)
        conn.close()
        print(f"[SERVER] thread for {client} stopped")


def main():
    if len(sys.argv) < 3:
        print("python server.py host:port authkey")
        return

    host, port = sys.argv[1].split(':')
    port = int(port)
    authkey = sys.argv[2].encode()

    stop_event = threading.Event()
    conn_threads: list[threading.Thread] = []

    listener_t = threading.Thread(
        target=listen_on_connections,
        args=(host, port, authkey, conn_threads, stop_event),
        daemon=True
    )
    listener_t.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[SERVER] shutting down...")
        stop_event.set()
        listener_t.join()
        for t in conn_threads:
            t.join()
        print("[SERVER] all threads stopped")


if __name__ == "__main__":
    main()
