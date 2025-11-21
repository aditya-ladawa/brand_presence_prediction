import os
import sys
import argparse
import time
import colorsys
import json

import cv2
import numpy as np
from ultralytics import YOLO
import torch


# ------------------ ARGUMENTS ------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    help='Path to YOLO model file, for example "all_models/large_model/train/weights/best.pt"',
    required=True,
)
parser.add_argument(
    "--source",
    help='Input video file path, for example "input_video.mp4"',
    required=True,
)
parser.add_argument(
    "--output",
    help='Output video file path, for example "output_video.mp4"',
    default="output_video.mp4",
)
parser.add_argument(
    "--thresh",
    help='Minimum confidence threshold, for example "0.4"',
    default=0.5,
)
parser.add_argument(
    "--resolution",
    help='Output resolution in WxH, for example "1920x1080"',
    default="1920x1080",
)
parser.add_argument(
    "--max_infer_size",
    help="Maximum image size used for YOLO inference (longer side)",
    type=int,
    default=960,
)

args = parser.parse_args()


# ------------------ PARSE ARGS ------------------

model_path = args.model
video_path = args.source
output_path = args.output
min_thresh = float(args.thresh)
res_str = args.resolution
MAX_INFER_SIZE = int(args.max_infer_size)

if "x" not in res_str:
    print('Resolution must be in "WxH" format, for example "1920x1080".')
    sys.exit(1)

outW, outH = map(int, res_str.split("x"))


# ------------------ LOAD MODEL ------------------

if not os.path.exists(model_path):
    print("ERROR: Model path is invalid or model was not found.")
    sys.exit(1)

model = YOLO(model_path, task="detect")
labels = model.names  # dict or list

# Device selection (you can force CPU by setting FORCE_DEVICE=cpu in env)
forced_device = os.environ.get("FORCE_DEVICE", "").lower()
if forced_device in ("cpu", "cuda"):
    device = forced_device
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda" and not torch.cuda.is_available():
    print("WARNING: CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
    device = "cpu"

model.to(device)
print(f"Using device: {device}")
print(f"Loaded model with {len(labels)} classes.")


# ------------------ OPEN VIDEO ------------------

if not os.path.isfile(video_path):
    print(f"ERROR: Input video file '{video_path}' not found.")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"ERROR: Could not open video '{video_path}'.")
    sys.exit(1)

# Try to get FPS from input video, fall back to 30 if invalid
input_fps = cap.get(cv2.CAP_PROP_FPS)
if input_fps is None or input_fps <= 1e-2:
    input_fps = 30.0

print(f"Input FPS: {input_fps:.2f}")

# MP4 writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_path,
    fourcc,
    input_fps,
    (outW, outH),
)

if not writer.isOpened():
    print("ERROR: Could not open VideoWriter. Try a different codec or check ffmpeg/gstreamer.")
    cap.release()
    sys.exit(1)


# ------------------ COLORS ------------------

def generate_colors(num_colors: int):
    """Generate a list of bright, distinct BGR colors."""
    colors = []
    for i in range(num_colors):
        hue = i / float(num_colors)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colors


# Handle labels as dict or list
if isinstance(labels, dict):
    class_ids = list(labels.keys())
    get_name = lambda cid: labels[cid]
else:
    class_ids = list(range(len(labels)))
    get_name = lambda cid: labels[cid]

num_classes = max(len(class_ids), 80)  # support many classes
bbox_colors = generate_colors(num_classes)


# ------------------ BRAND EXPOSURE STATS ------------------

# For each class id we track frames_visible and contiguous occurrences
brand_stats = {
    cid: {
        "name": get_name(cid),
        "frames_visible": 0,
        "total_occurrences": 0,
        "prev_present": False,
        # for visibility / position
        "sum_cx": 0.0,        # sum of box centers x (pixels)
        "sum_cy": 0.0,        # sum of box centers y (pixels)
        "sum_area": 0.0,      # sum of box areas (pixels)
    }
    for cid in class_ids
}

# ------------------ MAIN LOOP ------------------

avg_frame_rate = 0.0
frame_rate_buffer = []
fps_avg_len = 200
frame_index = 0

print("Starting inference on video...")

try:
    while True:
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Reached end of video or cannot read frame.")
            break

        # Resize to requested output resolution for display and writing
        frame = cv2.resize(frame, (outW, outH))
        orig_h, orig_w = frame.shape[:2]

        # Prepare frame for YOLO inference by resizing if needed
        inference_frame = frame
        scale_x = 1.0
        scale_y = 1.0

        max_dim = max(orig_w, orig_h)
        if max_dim > MAX_INFER_SIZE:
            scale = MAX_INFER_SIZE / max_dim
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            inference_frame = cv2.resize(frame, (new_w, new_h))
            scale_x = orig_w / float(new_w)
            scale_y = orig_h / float(new_h)

        # Run YOLO inference
        results = model(inference_frame, device=device, verbose=False)
        detections = results[0].boxes
        object_count = 0

        present_class_ids = set()

        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            # Scale boxes back to output frame size
            xmin = int(xmin * scale_x)
            xmax = int(xmax * scale_x)
            ymin = int(ymin * scale_y)
            ymax = int(ymax * scale_y)

            classidx = int(detections[i].cls.item())
            conf = detections[i].conf.item()

            if conf < min_thresh:
                continue

            classname = get_name(classidx)
            present_class_ids.add(classidx)

            # Center and area (in output frame coordinates)
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            area = float(max(0, xmax - xmin) * max(0, ymax - ymin))

            # Accumulate for visibility / position statistics
            if classidx in brand_stats:
                brand_stats[classidx]["sum_cx"] += cx
                brand_stats[classidx]["sum_cy"] += cy
                brand_stats[classidx]["sum_area"] += area

            color = bbox_colors[classidx % len(bbox_colors)]

            # Draw bounding box (thicker by +1)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)

            # Label text (larger font size)
            label = f"{classname}: {int(conf * 100)}%"
            label_scale = 0.7
            label_thickness = 2

            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness
            )
            label_ymin = max(ymin, label_size[1] + 10)

            cv2.rectangle(
                frame,
                (xmin, label_ymin - label_size[1] - 10),
                (xmin + label_size[0], label_ymin + base_line - 10),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                label,
                (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_scale,
                (0, 0, 0),
                label_thickness,
            )

            object_count += 1

        # Update exposure stats per class
        for cid, stats in brand_stats.items():
            present = cid in present_class_ids
            if present:
                stats["frames_visible"] += 1
            if present and not stats["prev_present"]:
                # new contiguous appearance
                stats["total_occurrences"] += 1
            stats["prev_present"] = present

        # FPS display (you can also bump font here if you like)
        cv2.putText(
            frame,
            f"FPS: {avg_frame_rate:0.2f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Objects: {object_count}",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # Write to output video
        writer.write(frame)

        # Optional preview on Ubuntu:
        # cv2.imshow("YOLO detection results", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        t_stop = time.perf_counter()
        frame_rate_calc = float(1.0 / max(t_stop - t_start, 1e-6))

        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
        avg_frame_rate = float(np.mean(frame_rate_buffer))

        frame_index += 1
        if frame_index % 50 == 0:
            print(f"Processed {frame_index} frames, current FPS: {avg_frame_rate:0.2f}")

except KeyboardInterrupt:
    print("Interrupted by user.")

# ------------------ CLEANUP ------------------

cap.release()
writer.release()

# In headless / non-GUI builds this can fail, so keep it safe
try:
    cv2.destroyAllWindows()
except cv2.error as e:
    print(f"cv2.destroyAllWindows() failed (no GUI backend). Ignoring. Details: {e}")

print(f"Average pipeline FPS: {avg_frame_rate:.2f}")
print(f"Output video saved to: {output_path}")



# ------------------ POSTPROCESS: BRAND EXPOSURE REPORT ------------------

video_duration_sec = frame_index / input_fps if input_fps > 0 else 0.0
video_duration_min = video_duration_sec / 60.0 if video_duration_sec > 0 else 0.0

brand_exposure = []

frame_area = float(outW * outH)

for cid, stats in brand_stats.items():
    if stats["frames_visible"] == 0:
        continue  # never visible, skip

    frames_visible = stats["frames_visible"]
    total_occ = stats["total_occurrences"]

    total_duration_sec = frames_visible / input_fps
    avg_duration_per_occ = (
        total_duration_sec / total_occ if total_occ > 0 else 0.0
    )
    freq_per_min = (
        total_occ / video_duration_min if video_duration_min > 0 else 0.0
    )

    # Average position and area when visible
    avg_cx = stats["sum_cx"] / frames_visible
    avg_cy = stats["sum_cy"] / frames_visible
    avg_cx_norm = avg_cx / outW   # 0-1
    avg_cy_norm = avg_cy / outH   # 0-1

    # Simple position zone: left/center/right and top/middle/bottom
    def horiz_zone(x_norm: float) -> str:
        if x_norm < 1.0 / 3.0:
            return "left"
        elif x_norm < 2.0 / 3.0:
            return "center"
        else:
            return "right"

    def vert_zone(y_norm: float) -> str:
        if y_norm < 1.0 / 3.0:
            return "top"
        elif y_norm < 2.0 / 3.0:
            return "middle"
        else:
            return "bottom"

    position_zone = f"{vert_zone(avg_cy_norm)}-{horiz_zone(avg_cx_norm)}"

    # Visibility as area
    total_area = stats["sum_area"]                # sum over all frames where brand appears
    # average relative area when the brand is visible
    avg_rel_area_visible = (
        (total_area / frames_visible) / frame_area if frames_visible > 0 else 0.0
    )
    # relative area over the whole match (including frames where it is absent)
    avg_rel_area_global = (
        total_area / (frame_area * frame_index) if frame_index > 0 else 0.0
    )

    brand_exposure.append(
        {
            "class_id": int(cid),
            "name": stats["name"],
            "frames_visible": int(frames_visible),
            "total_occurrences": int(total_occ),
            "total_duration_seconds": float(total_duration_sec),
            "avg_duration_per_occurrence_seconds": float(avg_duration_per_occ),
            "occurrences_per_minute": float(freq_per_min),
            # new fields
            "avg_center_x_normalized": float(avg_cx_norm),
            "avg_center_y_normalized": float(avg_cy_norm),
            "position_zone": position_zone,
            "avg_relative_area_when_visible": float(avg_rel_area_visible),
            "avg_relative_area_overall": float(avg_rel_area_global),
        }
    )


# Sort by total visible duration (descending) to get "most exposed" brands
brand_exposure.sort(
    key=lambda x: x["total_duration_seconds"], reverse=True
)

top_15 = brand_exposure[:15]

# Build JSON summary
stats_json = {
    "video": {
        "path": video_path,
        "fps": float(input_fps),
        "total_frames": int(frame_index),
        "duration_seconds": float(video_duration_sec),
        "duration_minutes": float(video_duration_min),
    },
    "brands": brand_exposure,
    "top_15_by_total_duration": [b["name"] for b in top_15],
}

base, ext = os.path.splitext(output_path)
json_path = base + "_stats.json"
md_path = base + "_stats.md"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(stats_json, f, indent=2)

print(f"Exposure stats JSON saved to: {json_path}")

# Build Markdown report
def format_seconds(sec: float) -> str:
    return f"{sec:.2f}"

md_lines = []
md_lines.append("# Brand Exposure Report\n")
md_lines.append(f"- Video: `{video_path}`")
md_lines.append(f"- Duration: {video_duration_min:.2f} minutes ({video_duration_sec:.2f} seconds)")
md_lines.append(f"- FPS: {input_fps:.2f}")
md_lines.append(f"- Total frames: {frame_index}")
md_lines.append("")
md_lines.append("## Top 15 brands by total visible duration\n")
md_lines.append("| Rank | Brand | Total duration (s) | Occurrences | Avg duration / occurrence (s) | Occurrences / minute | Avg relative area (visible) | Position zone |")
md_lines.append("|------|-------|--------------------|-------------|--------------------------------|----------------------|-----------------------------|---------------|")

for idx, b in enumerate(top_15, start=1):
    md_lines.append(
        f"| {idx} | {b['name']} | {format_seconds(b['total_duration_seconds'])} | "
        f"{b['total_occurrences']} | {format_seconds(b['avg_duration_per_occurrence_seconds'])} | "
        f"{b['occurrences_per_minute']:.2f} | "
        f"{b['avg_relative_area_when_visible']*100:.2f}% | {b['position_zone']} |"
    )


md_lines.append("")
md_lines.append("## All detected brands (sorted by total duration)\n")
md_lines.append("| Brand | Total duration (s) | Frames visible | Occurrences | Avg duration / occurrence (s) | Occurrences / minute |")
md_lines.append("|-------|--------------------|----------------|-------------|--------------------------------|----------------------|")

for b in brand_exposure:
    md_lines.append(
        f"| {b['name']} | {format_seconds(b['total_duration_seconds'])} | {b['frames_visible']} | "
        f"{b['total_occurrences']} | {format_seconds(b['avg_duration_per_occurrence_seconds'])} | "
        f"{b['occurrences_per_minute']:.2f} |"
    )

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"Exposure stats Markdown saved to: {md_path}")
