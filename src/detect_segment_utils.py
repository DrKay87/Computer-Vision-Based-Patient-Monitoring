import cv2
import numpy as np

def Kmeans_segment(image, k):

    non_zero_mask = image > 0  
    pixels = image[non_zero_mask].reshape(-1, 1).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    sorted_indices = np.argsort(centers.flatten())
    correct_labels = np.zeros_like(labels)

    correct_labels[labels == sorted_indices[1]] = 1  
    correct_labels[labels == sorted_indices[0]] = 0 

    mask = np.zeros_like(image, dtype=np.uint8)
    mask[non_zero_mask] = correct_labels.flatten()

    return mask

def dilate_mask(mask,kernel_size):

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return mask

def find_empty_regions(mask):

    mask = np.uint8(mask > 0) * 255  

    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    num_holes = 0
    for contour in contours:
        if cv2.contourArea(contour) > 0: 
            num_holes += 1
    
    return num_holes - 1


def fill_small_holes(mask, k=5):

    kernel = np.ones((k, k), np.uint8)
    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return filled_mask

def fill_empty_regions(mask):

    mask = np.uint8(mask > 0) * 255  
    previous = 0
    count = 0
    kernel_size = 15
    while True: 

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is not None:
            for i in range(len(contours)):

                if hierarchy[0][i][3] != -1:
                    cv2.drawContours(mask, [contours[i]], -1, 255, thickness=cv2.FILLED)
                    fill_small_holes(mask, k=15)
        else:
            break
        
        if len(contours) == previous:

            mask = fill_small_holes(mask,kernel_size)
            count = count + 1
            kernel_size = kernel_size + 5
            
            if count == 5:
                break

        previous = len(contours)

    mask = mask/255

    return mask


def Detect_roi(image):
    
    k = 2
    mask = Kmeans_segment(image, k)
    
    kernel_size = 3
    mask = dilate_mask(mask,kernel_size)
    mask = fill_empty_regions(mask)
    
    return np.array(mask)


def patient_activity_mask(video_path, patient_box, start=100, end=6100):
    cap = cv2.VideoCapture(video_path)
    x1, y1, x2, y2 = map(int, patient_box)
    H, W = y2-y1,x2-x1
    motion_count = np.zeros((H, W))
    current_idx = 0
    prev = None
    while True:
        ret, frame = cap.read()
        if not ret and current_idx > end:
            break
    
        if current_idx >= start:
            x1, y1, x2, y2 = map(int, patient_box)
            frame_roi = frame[y1:y2, x1:x2]

            if prev is None:
                prev = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
                continue

            curr = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(curr, prev)
            motion_count = motion_count + diff

            prev = curr

        current_idx += 1
        
    motion_estimate = cv2.GaussianBlur(motion_count, (5, 5), 0)
    motion_mask = Detect_roi(motion_estimate)
    min_size = 2000
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.int8(motion_mask*255), connectivity=8)
    motion_mask = np.zeros_like(motion_mask)
    for i in range(1, num_labels):  
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            motion_mask[labels == i] = 1
    #largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    #motion_mask[labels == largest_label] = 1

    return motion_estimate, motion_mask

def PCA_angle(mask):
    coords = np.column_stack(np.where(mask > 0))
    coords = coords[:, ::-1]
    pts_orig = coords.astype(np.float32)
    center = pts_orig.mean(axis=0)
    pts_normalized = pts_orig - center
    _, _, Vt = np.linalg.svd(pts_normalized, full_matrices=False)
    direction = Vt[0]
    angle = np.arctan2(direction[1], direction[0])
    
    return center, angle

def rotate_img(img,angle,center):
    angle_deg = np.degrees(angle)
    cx, cy = center
    H, W = img.shape[:2]
    R = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0) 
    rotated_img = cv2.warpAffine( img, R, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return rotated_img

def find_bed_rect(bed_mask):

    rot_coords = np.column_stack(np.where(bed_mask > 0))
    rot_coords = rot_coords[:, ::-1]  # (y,x) → (x,y)
    x = rot_coords[:, 0]
    y = rot_coords[:, 1]
    
    bed_x1 = int(np.percentile(x, 5))
    bed_x2 = int(np.percentile(x, 95))
    
    bed_y1 = int(np.percentile(y, 0))
    bed_y2 = int(np.percentile(y, 100))

    return [bed_x1, bed_y1, bed_x2, bed_y2]

def torso_roi_from_upperbody(box,orient=1):
    
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    if orient is 0:
        rx1 = x1 + 0.1 * w
        rx2 = x2 - 0.1*w
        ry1 = y1 + 0.2 * h
        ry2 = y2 - 0.1*h
    else:
        rx1 = x1 + 0.1 * w
        rx2 = x2 - 0.1*w
        ry1 = y1 + 0.1 * h
        ry2 = y2 - 0.2*h
    
    return int(rx1), int(ry1), int(rx2), int(ry2)

def detect_torso_orientation_and_adjust_bed_center(motion,rect):
    [x1,y1,x2,y2] = rect 
    motion = motion[y1:y2,x1:x2]
    h, w = y2-y1, x2-x1
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    total_mass = np.sum(motion) + 1e-6
    cx = np.sum(x_coords * motion) / total_mass
    cy = np.sum(y_coords * motion) / total_mass

    offset_x = cx-w//2
    adjusted_rect = [int(x1+offset_x),int(y1),int(x2+offset_x),int(y2)]


    if cy < h / 2:
        upper_body_region = [int(x1+offset_x),int(y1),int(x2+offset_x),int(h//2)]
        torso_region = torso_roi_from_upperbody(upper_body_region,orient=0)
    else:
        upper_body_region = [int(x1+offset_x),int(h//2),int(x2+offset_x),int(y2)]
        torso_region = torso_roi_from_upperbody(upper_body_region,orient=1)

    
    return torso_region, adjusted_rect