
import cv2
import numpy as np
import random
from DetectnSegment_Utils import rotate_img
from vmdpy import VMD

def smooth_signal(x, window=30):
    return np.convolve(x, np.ones(window)/window, mode='same')

def center_of_mass(mag):
    h, w = mag.shape
    ys, xs = np.mgrid[0:h, 0:w]

    motion_sum = np.sum(mag)

    if motion_sum > 1e-6:
        cx = np.sum(xs * mag) / motion_sum
        cy = np.sum(ys * mag) / motion_sum
    else:
        cx, cy = np.nan, np.nan
    return cx, cy

def extract_respiration_signal(video_path, patient_box, bed_box, torso_box, angle_rot, center_rot, flow_scale=1):
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    resp_signal = []
    movement = []
    motion_centroids = []
    
    [x1, y1, x2, y2] = map(int, patient_box)
    [bed_x1, bed_y1, bed_x2, bed_y2] = map(int, bed_box)
    [tors_x1, tors_y1, tors_x2, tors_y2] = map(int, torso_box)

    prev_gray = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_roi = frame[y1:y2, x1:x2]
        rotated_frame_roi = rotate_img(frame_roi,angle_rot,center_rot)
        bed_region = rotated_frame_roi[bed_y1:bed_y2,bed_x1:bed_x2]
   
        image = bed_region
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

     
        if flow_scale != 1.0:
            gray = cv2.resize(gray, None, fx=flow_scale, fy=flow_scale,
                              interpolation=cv2.INTER_AREA)

        if prev_gray is None:
            prev_gray = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        dy = flow[..., 1]
        dy1 = dy[tors_y1-bed_y1:tors_y2-bed_y1,tors_x1-bed_x1:tors_x2-bed_x1]
        
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        cx,cy = center_of_mass(mag)

        motion_centroids.append((cx+bed_x1, cy+bed_y1))
        resp_signal.append(np.median(dy1))
        movement.append(np.mean(mag))
        
        prev_gray = gray

    cap.release()

    return np.array(resp_signal), np.array(movement), np.array(motion_centroids)

def has_artifacts(d, ratio_thresh=8.0):
  
    typical = np.percentile(d, 80)
    extreme = np.percentile(d, 99)

    if typical < 1e-6:
        return extreme > 0   

    return (extreme / typical) > ratio_thresh

def adaptive_derivative_artifact_removal(signal, der_pct=99.5,ratio_thresh=5.0):
    sig = signal.copy()
    d = np.abs(np.diff(sig, prepend=sig[0]))

    while has_artifacts(d, ratio_thresh):

        d = np.abs(np.diff(sig, prepend=sig[0]))
        thresh = np.percentile(d, der_pct)
        artifact_idx = d > thresh
        sig[artifact_idx] = 0
    return sig, artifact_idx

def apply_vmd(resp_sig, K = 10, alpha = 2000, tau = 0., DC = 0, init = 1, tol = 1e-7):
    #alpha = 2000 # moderate bandwidth constraint 
    #tau = 0. # noise-tolerance (0 for no strict fidelity) 
    #K = 5 # number of modes to decompose into 
    #DC = 0 # no DC component 
    #init = 1 # initialize omegas uniformly 
    #tol = 1e-7 
    # ----------------------------- # Run VMD # ----------------------------- 
    u, u_hat, omega = VMD(resp_sig, alpha, tau, K, DC, init, tol)

    return u, u_hat, omega

def actual_bed_boundaries(box):
    
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    rx1 = x1 + 0.2 * w
    rx2 = x2 - 0.2*w
    ry1 = y1 + 0.05 * h
    ry2 = y2 - 0.05*h

    return int(rx1), int(ry1), int(rx2), int(ry2)