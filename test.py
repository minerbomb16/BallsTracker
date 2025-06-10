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