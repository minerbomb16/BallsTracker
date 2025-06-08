import cv2

def list_cameras(max_index=10, backend=None):
    """
    Probe camera indices [0..max_index-1] and return the ones that open successfully.
    
    Args:
      max_index (int): how many indices to try (0 through max_index-1).
      backend (int, optional): OpenCV capture backend, e.g. cv2.CAP_DSHOW (Windows),
                               cv2.CAP_V4L2 (Linux), cv2.CAP_AVFOUNDATION (macOS).
                               If None, lets OpenCV choose the default.
    
    Returns:
      List[int]: indices of available cameras.
    """
    available = []
    for idx in range(max_index):
        # try to open the capture device
        if backend is not None:
            cap = cv2.VideoCapture(idx, backend)
        else:
            cap = cv2.VideoCapture(idx)
        
        if not cap.isOpened():
            cap.release()
            continue
        
        # optional: try to grab a frame to be extra sure
        ret, _ = cap.read()
        cap.release()
        if ret:
            available.append(idx)
    return available

if __name__ == "__main__":
    # On Windows you might pass cv2.CAP_DSHOW; on Linux cv2.CAP_V4L2; on Mac cv2.CAP_AVFOUNDATION
    cams = list_cameras(max_index=8, backend=None)
    if cams:
        print(f"Found camera device indices: {cams}")
    else:
        print("No camera devices detected.")

