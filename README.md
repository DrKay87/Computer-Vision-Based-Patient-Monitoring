# Patient Movement and Breathing Signal Detection from Video

This repository contains a computer-vision pipeline for detecting patient movement, estimating breathing-related motion signals, and deriving simple sleep/movement indicators from video recordings.

The current implementation combines person detection, patient-region tracking, bed/activity region estimation, optical-flow-based torso motion extraction, artifact removal, variational mode decomposition, respiration-rate estimation, sleep/wake proxy metrics, and bed-edge movement flags.

## Main features

- Detects the patient/person region in video frames using YOLOv8.
- Tracks patient-related detections across frames and merges overlapping patient boxes.
- Estimates the active patient/bed region using motion accumulation and segmentation.
- Aligns the bed/patient region using PCA-based rotation.
- Extracts torso motion using Farneback optical flow.
- Derives a respiration-related signal from vertical torso motion.
- Removes large derivative artifacts from the raw respiration signal.
- Applies variational mode decomposition to isolate breathing-related components.
- Estimates respiration rate in breaths per minute.
- Computes movement-based sleep/wake proxy metrics.
- Flags when the patient movement centroid is close to the bed edge.

## Repository structure

```text
patient-movement-breathing-signal-repo/
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── .gitignore
├── src/
│   ├── run_pipeline.py
│   ├── detect_segment_utils.py
│   ├── signal_detection_utils.py
│   └── utilities.py
├── data/
│   └── README.md
├── outputs/
│   └── README.md
└── docs/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/patient-movement-breathing-signal-detection.git
cd patient-movement-breathing-signal-detection
```

Create a Python environment:

```bash
conda env create -f environment.yml
conda activate patient-motion-signals
```

Or install with pip:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Place your input video in `data/raw/`.
2. Open `src/run_pipeline.py`.
3. Change the `video_path` variable to point to your video file.
4. Run:

```bash
python src/run_pipeline.py
```

The script will process the video and save output figures such as:

```text
Respiration.png
Sleep_metrics.png
Bed_fall_Flags.png
```

You may want to update the script so these files are saved directly into `outputs/`.

## Pipeline overview

### 1. Person detection and tracking

YOLOv8 is used to detect the person class in each frame. Detected boxes are linked across frames using center-distance matching. Persistent tracks are summarized using presence, mean area, aspect ratio, and vertical stability.

### 2. Patient-region selection

The most persistent and spatially dominant person track is selected as the reference patient ID. Other tracks overlapping with the reference patient region are merged to create a robust patient bounding box.

### 3. Bed and activity-region detection

Motion is accumulated inside the patient region. K-means segmentation and morphological post-processing are used to estimate the patient/bed activity mask. PCA is then used to rotate the region and align the bed with the image axes.

### 4. Breathing signal extraction

Farneback optical flow is computed between consecutive frames. The median vertical optical-flow component inside the torso region is used as the raw breathing-related signal.

### 5. Signal cleaning and respiration estimation

Large derivative artifacts are removed adaptively. Variational mode decomposition is then applied to separate signal modes, and peaks in the filtered breathing signal are used to estimate respiration rate.

### 6. Sleep and bed-edge indicators

Movement magnitude and respiration variability are aggregated into one-minute epochs to derive a simple sleep/wake proxy. Motion centroids are also compared with estimated bed boundaries to flag near-edge movement.

## Input requirements

- A video file showing the patient in bed.
- The patient should be visible for a sufficient portion of the video.
- The current pipeline assumes a mostly fixed camera view.
- For privacy reasons, do not upload or commit identifiable patient videos.

## Important privacy note

This repository is intended for research and prototyping. Patient videos and clinical data should not be committed to GitHub. Use anonymized or synthetic demo data when sharing the repository publicly.

## Current limitations

- The video path is currently hard-coded in `src/run_pipeline.py` and should be replaced with command-line arguments in a future version.
- The pipeline uses heuristic thresholds for tracking, sleep/wake proxy detection, artifact removal, and bed-edge detection.
- The current sleep/wake output should be interpreted as a movement-based proxy, not a clinically validated sleep-stage classifier.
- Performance depends on camera position, lighting/infrared quality, bedding occlusion, patient posture, and video frame rate.

## Recommended next improvements

- Add command-line arguments using `argparse`.
- Save all generated plots and signal arrays to `outputs/`.
- Add a short anonymized demo video or synthetic example.
- Add unit tests for bounding-box utilities and signal-processing functions.
- Add a notebook in `docs/` explaining the full workflow with figures.
- Package the code as an installable Python module.

## Citation

If you use this repository, please cite the associated paper, thesis, or project report. You can also update `CITATION.cff` with the final publication details.

## License

This project is released under the MIT License. Update the license if your institution, collaborators, or data-sharing agreements require another license.
