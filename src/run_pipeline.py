import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from pathlib import Path
from Utilities import center_distance, iou, compute_union_box, smooth_boxes, expand_box
from DetectnSegment_Utils import patient_activity_mask, detect_torso_orientation_and_adjust_bed_center, find_bed_rect, rotate_img, PCA_angle
from SigDetect_Utils import center_of_mass, extract_respiration_signal, adaptive_derivative_artifact_removal, apply_vmd, smooth_signal, actual_bed_boundaries
from collections import defaultdict
from ultralytics import YOLO

from scipy.signal import find_peaks
from vmdpy import VMD

print('Loading YOLO ....')

model = YOLO("yolov8l.pt")  # small & fast is enough
model.eval()

######################### Read Video ##################
video_path = r"C:\Users\au711969\OneDrive - Aarhus universitet\My_Task\health-sensing-challenge\health-sensing\night_reo_ir_on_amcrest_br.mp4"
#video_path = r"C:\Users\au711969\OneDrive - Aarhus universitet\My_Task\health-sensing-challenge\health-sensing\in_bed\data\in_bed.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_s = n_frames / fps if fps > 0 else None

######################### Apply YOLO ##################

detections_per_frame = []

img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print('Applying YOLOv8L on video frames and detecting person specific class .... ')
print('This may take a while')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=2560, conf=0.2, iou=0.5, verbose=False)

    frame_dets = []
    for r in results:
        if r.boxes is None:
            continue

        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if int(cls) == 0:  # person
                x1, y1, x2, y2 = map(float, box.tolist())
                # Clip to frame
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))

                frame_dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(conf)
                })

        detections_per_frame.append(frame_dets)

print('Detection of Persons through YOLOv8L is complete complete now...')
################################# Track Boxes ######################################
print('Now moving to tracking of Patient ID')
class Track:
    def __init__(self, tid, bbox):
        self.id = tid
        self.bbox = bbox
        self.missed = 0

MAX_DIST = 120   
MAX_MISSED = 20  

tracks = []
next_track_id = 0
tracks_per_frame = []

for frame_idx, detections in enumerate(detections_per_frame):

    assigned = set()

    for track in tracks:
        best_det = None
        best_dist = float("inf")

        for d_idx, det in enumerate(detections):
            if d_idx in assigned:
                continue
            dist = center_distance(track.bbox, det["bbox"])
            if dist < best_dist:
                best_dist = dist
                best_det = d_idx

        if best_det is not None and best_dist < MAX_DIST:
            track.bbox = detections[best_det]["bbox"]
            track.missed = 0
            assigned.add(best_det)
        else:
            track.missed += 1

    for d_idx, det in enumerate(detections):
        if d_idx not in assigned:
            tracks.append(Track(next_track_id, det["bbox"]))
            next_track_id += 1

    tracks = [t for t in tracks if t.missed <= MAX_MISSED]

    frame_tracks = [{"id": t.id, "bbox": t.bbox} for t in tracks]
    tracks_per_frame.append(frame_tracks)

##################### Tracking Statistics ###############

track_stats = defaultdict(lambda: {
    "frames": 0,
    "areas": [],
    "centers": [],
    "aspect_ratios": []
})

for frame_tracks in tracks_per_frame:
    for t in frame_tracks:
        tid = t["id"]
        x1, y1, x2, y2 = t["bbox"]

        w = x2 - x1
        h = y2 - y1
        area = w * h
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        track_stats[tid]["frames"] += 1
        track_stats[tid]["areas"].append(area)
        track_stats[tid]["centers"].append((cx, cy))
        track_stats[tid]["aspect_ratios"].append(w / (h + 1e-6))

track_features = {}

for tid, s in track_stats.items():
    centers = np.array(s["centers"])
    areas = np.array(s["areas"])

    track_features[tid] = {
        "presence": s["frames"],
        "mean_area": areas.mean(),
        "area_std": areas.std(),
        "center_std_y": centers[:,1].std(),
        "mean_ar": np.mean(s["aspect_ratios"]),
        "mean_y": centers[:,1].mean()
    }

TOTAL_FRAMES = len(tracks_per_frame)

persistent_ids = [
    tid for tid, f in track_features.items()
    if f["presence"] > 0.01 * TOTAL_FRAMES
]

max_area = max(f["mean_area"] for f in track_features.values())

bed_candidates = []

ref_patient_id = max(
    persistent_ids,
    key=lambda tid: track_features[tid]["mean_area"]
)

print("####################### Persistent ID detection statistics ################################")
for tid in persistent_ids:
    f = track_features[tid]
    print(
        f"ID {tid}: "
        f"presence={f['presence']}, "
        f"mean_area={f['mean_area']:.0f}, "
        f"mean_ar={f['mean_ar']:.2f}, "
        f"center_std_y={f['center_std_y']:.2f}"
    )
print(" ")

print("####################### Selected Patient ID reference ################################")
print(" ")
print("Selected reference patient ID:", ref_patient_id)

################# Detect Patient Region ####################################
print('.')
print('.')
print('.')
print('Now detecting person boxes overlapping with reference patient box')

core_id = ref_patient_id 

core_boxes = []
for frame_tracks in tracks_per_frame:
    for t in frame_tracks:
        if t["id"] == core_id:
            core_boxes.append(t["bbox"])

core_boxes = np.array(core_boxes)
core_ref_box = np.median(core_boxes, axis=0)

################### Track all boxes overlapping with patient box ##############

track_overlap = {}

for tid, s in track_stats.items():
    overlaps = []

    for frame_tracks in tracks_per_frame:
        for t in frame_tracks:
            if t["id"] == tid:
                overlaps.append(iou(t["bbox"], core_ref_box))

    if overlaps:
        track_overlap[tid] = np.mean(overlaps)
patient_related_ids = [
    tid for tid, ov in track_overlap.items()
    if ov > 0.05   
]
print("Detected {len(patient_related_ids)} Patient-related track IDs")
print('.')
print('.')
print('.')
print("Now merging Patient-related track IDs")

################### Merge all boxes overlapping with patient box ##############

merged_patient_boxes = []

for frame_tracks in tracks_per_frame:
    boxes = [
        t["bbox"] for t in frame_tracks
        if t["id"] in patient_related_ids
    ]

    if len(boxes) == 0:
        merged_patient_boxes.append(None)
        continue

    [x1, y1, x2, y2] = compute_union_box(boxes)
    

    merged_patient_boxes.append([x1, y1, x2, y2])

total_frames = len(merged_patient_boxes)
print("All Patient related IDs Merged")

frames_with_box = sum(
    box is not None for box in merged_patient_boxes
)

print("Total number of frames with patient box:", frames_with_box)

coverage = 100.0 * frames_with_box / total_frames
print(f"Percentage of Coverage: {coverage:.2f}%")

print('.')
print('.')
print('.')
print("Now smoothing the patient boxes and finding one union region")

smoothed_merged_patient_boxes = smooth_boxes(merged_patient_boxes, core_ref_box)


ref_box = compute_union_box(smoothed_merged_patient_boxes)

exp_ref_box = expand_box(ref_box, img_w, img_h, scale_x=0.25, scale_y=0.25)

print('.')
print('.')
print('.')
print("Now detecting patient activity region to detect bed area")
################ Patient Activity Masking ###############################

total_time = n_frames/fps/60
start_time = 1
end_time = 9
if end_time <= total_time:
    end_time = start_time + end_time
else:
    end_time = total_time

start = start_time*60*10+1
end = end_time*60*10
motion_estimate, motion_mask = patient_activity_mask(video_path, ref_box, start=100, end=6100)

################ Rotate Image to align with image coordinates #############

center_rot, angle_rot = PCA_angle(motion_mask)
rotated_mask = rotate_img(motion_mask, angle_rot, center_rot)
rotated_motion = rotate_img(motion_estimate, angle_rot, center_rot)

[bed_x1, bed_y1, bed_x2, bed_y2] = find_bed_rect(rotated_mask)
bed_rect = [bed_x1, bed_y1, bed_x2, bed_y2]

torso_region, bed_rect_new = detect_torso_orientation_and_adjust_bed_center(rotated_motion, bed_rect)

[bed_x1, bed_y1, bed_x2, bed_y2] = bed_rect_new
[torsx1, torsy1, torsx2, torsy2] = torso_region

print('.')
print('.')
print('.')
print("Now computing optical flow based respiration and motion signals from detected torso and patient regions")
###################### Compute movement related metrics using optical flow #####################

bed_box = bed_rect_new
torso_box = torso_region
patient_box = ref_box

resp_raw, movement_profile, motion_centroids = extract_respiration_signal(video_path, patient_box, bed_box, torso_box, angle_rot, center_rot, flow_scale=1)

print('.')
print('.')
print('.')
print("Now plotting the respiration, sleep and bed-edge related signals")
###################### Detect Respiration Signal #####################

resp_sig = resp_raw
sig = resp_sig.copy()
d = np.abs(np.diff(sig, prepend=sig[0]))

without_artifacts, artifact_idx = adaptive_derivative_artifact_removal(resp_sig, der_pct=95,ratio_thresh=5.0)
u, u_hat, omega = apply_vmd(without_artifacts*100)
#resp_filt = np.sum(u[0:1, :], axis=0)/100
resp_filt = u[0, :]/100

fs = float(fps)               
offset = 100                  
window_sec = 10               
window_samples = int(window_sec * fs)

every_minute_bpm = []
every_minute_taxis = []

prev_idx = offset

while prev_idx + window_samples <= n_frames:
    idx = prev_idx + window_samples

    if idx < 60*fps:
        start = offset
    else:
        start = idx - 60*fps

    segment = resp_filt[prev_idx:idx]

    peaks, _ = find_peaks(
        segment,
        distance=int(2 * fs)   # ≥2 sec between breaths
    )

    breaths = len(peaks)
    bpm = breaths * 60 / window_sec

    every_minute_bpm.append(bpm)
    every_minute_taxis.append(prev_idx / fs / 60)  # minutes
    prev_idx = idx
    
every_minute_bpm = np.array(every_minute_bpm)
every_minute_taxis = np.array(every_minute_taxis)


peaks, _ = find_peaks(resp_filt, distance=fs*2)  
rr_bpm = len(peaks) / (len(resp_filt)/fs) * 60

print("Overall Respiration rate (bpm):", rr_bpm)

t_axis = np.arange(n_frames-1) / fps / 60
plt.figure(figsize=(15,15))
plt.subplot(4,1,1)
plt.plot(t_axis,resp_sig)
plt.title("Measured movements of torso region")
plt.xlabel("Minutes")
plt.ylabel("Torso activity")

plt.subplot(4,1,2)
plt.plot(t_axis,d)
plt.title("drivative")
plt.xlabel("Minutes")
plt.ylabel("Torso activity")

plt.subplot(4,1,3)
plt.plot(t_axis,without_artifacts)
plt.plot(t_axis,resp_filt)
plt.title("Respiration related signal")
plt.xlabel("Minutes")
plt.ylabel("Respiration Signal")

plt.subplot(4,1,4)
plt.stem(every_minute_taxis,every_minute_bpm)
plt.title("Respiration rate")
plt.xlabel("Minutes")
plt.ylabel("bpm")
plt.savefig('Respiration.png')
plt.close()

###################### Detect Sleep Cycles #####################

fs = int(fps)
movement_smooth = smooth_signal(movement_profile, window=3 * fs)  # ~3 seconds

EPOCH_SEC = 60
EPOCH = fs * EPOCH_SEC

def epoch_mean(x, epoch):
    n = len(x) // epoch
    return np.array([
        np.mean(x[i*epoch:(i+1)*epoch])
        for i in range(n)
    ])

movement_epoch = epoch_mean(movement_smooth, EPOCH)
resp_epoch = epoch_mean(resp_filt, EPOCH)

THRESH = np.percentile(movement_epoch, 50)
#sleep_mask = movement_epoch < THRESH

def enforce_min_duration(mask, min_len):
    out = mask.copy()
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i < min_len:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out

def respiration_variability(resp_segment, fs):
    peaks, _ = find_peaks(resp_segment, distance=fs*2)
    if len(peaks) < 3:
        return np.nan
    ibi = np.diff(peaks) / fs
    return np.std(ibi)

resp_var_epoch = []

for i in range(len(resp_epoch)):
    start = i * EPOCH
    end = start + EPOCH
    if end > len(resp_filt):
        break
    resp_var_epoch.append(
        respiration_variability(resp_filt[start:end], fs)
    )

resp_var_epoch = np.array(resp_var_epoch)
RESP_THR = np.nanpercentile(resp_var_epoch, 50)  # median
resp_var_trend = smooth_signal(resp_var_epoch, window=10)  # ~10 minutes
sleep_mask = (movement_epoch < THRESH) & (resp_var_epoch < RESP_THR)
sleep_mask = enforce_min_duration(sleep_mask, min_len=2)  # ≥5 minutes

total_sleep_minutes = np.sum(sleep_mask)
time_in_bed_minutes = len(sleep_mask)
sleep_efficiency = total_sleep_minutes / time_in_bed_minutes

awakenings = np.sum(
    (sleep_mask[:-1] == True) & (sleep_mask[1:] == False)
)

plt.figure(figsize=(14, 8))

plt.subplot(4, 1, 1)
plt.plot(t_axis,resp_filt)
plt.title("Filtered Respiration Signal")
plt.ylabel("Amplitude")
plt.xlabel('Minutes')

plt.subplot(4, 1, 2)
plt.plot(movement_epoch)
plt.axhline(THRESH, color="r", linestyle="--")
plt.title("Movement (1-min epochs)")
plt.ylabel("Motion")

plt.subplot(4, 1, 3)
plt.plot(sleep_mask.astype(int))
plt.title("Sleep / Wake (1 = Sleep)")
plt.ylabel("State")

plt.subplot(4, 1, 4)
plt.plot(resp_var_epoch)
plt.title("Respiration Variability (Sleep Quality Proxy)")
plt.ylabel("IBI Std")

plt.xlabel("Minutes")
plt.tight_layout()
plt.savefig('Sleep_metrics')

################################# Bed fall allert ###############

[bed_x1, bed_y1, bed_x2, bed_y2] = actual_bed_boundaries(bed_rect_new)

bed_h = bed_y2 - bed_y1
bed_w = bed_x2 - bed_x1
edge_margin = 0.15 * bed_w  
bed_left_edge  = bed_x1 + edge_margin
bed_right_edge = bed_x1 + bed_w - edge_margin
edge_flags = []

for c in motion_centroids:
    if c is None or np.isnan(c[0]):
        edge_flags.append(False)
        continue

    cx, cy = c

    near_edge = (cx <= bed_left_edge) or (cx >= bed_right_edge)
    edge_flags.append(near_edge)

edge_flags = np.array(edge_flags, dtype=np.float32)

window = int(3 * fs)  # 3 seconds
edge_smooth = np.convolve(
    edge_flags,
    np.ones(window)/window,
    mode="same"
)
edge_state = edge_smooth > 0.7
cx_list = [c[0] if c is not None else np.nan for c in motion_centroids]
cx_list_t_axis = np.array(range(len(cx_list)))/fps/60.0

plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
plt.plot(cx_list_t_axis,cx_list, label="motion centroid x")
plt.axhline(bed_left_edge, color="r", linestyle="--", label="left edge")
plt.axhline(bed_right_edge, color="g", linestyle="--", label="right edge")
plt.legend()
plt.title("Motion centroid inside bed")
plt.ylabel('Bed edges')
plt.xlabel('Minutes')

plt.subplot(2,1,2)
plt.plot(cx_list_t_axis,edge_state.astype(int))
plt.title("Edge state (1 = near bed edge)")
plt.ylabel('Flag')
plt.xlabel('Minutes')
plt.savefig('Bed_fall_Flags.png')

