# Corridor Object Detection

This project explores the use of object detection to identify items placed along residential corridors from recorded videos.

The aim is to reduce the amount of manual video review needed by automatically extracting useful frames, detecting objects, and eventually generating reports of items found in each video.

## Current Workflow

The current workflow is:

```text
Corridor videos
      ↓
Extract useful frames
      ↓
Remove blurry / similar frames
      ↓
Create dataset
      ↓
Label corridor items
      ↓
Train custom YOLO model
      ↓
Run detection on videos
      ↓
Generate annotated videos + reports
