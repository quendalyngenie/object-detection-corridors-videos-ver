from ultralytics import YOLOE
from pathlib import Path
from collections import defaultdict
import csv
import torch


# settings

VIDEO_FOLDER = Path("videos")

# larger model = better detection, but slower
MODEL_NAME = "yoloe-26m-seg.pt"

# keep this low for video
# weak detections are allowed into the tracker
# we filter them properly later
CONFIDENCE = 0.20

# higher resolution helps detect smaller objects
IMAGE_SIZE = 640

# 1 = analyse every frame
# 2 = every second frame
VIDEO_STRIDE = 1

# maximum detections to prevent over prediction
# this only limits how many detections YOLO can return
MAX_DETECTIONS = 30

# object must appear for at least this many frames
# before we accept it as a real tracked object
MIN_TRACK_FRAMES = 3

# object must also have at least this many
# detections above its class confidence threshold
MIN_STRONG_FRAMES = 2

# botsort is better if camera is moving
# eg walking through the corridor
TRACKER = "bytetrack.yaml"


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv"
}


# device

DEVICE = "cpu"

# if torch.backends.mps.is_available():
#     DEVICE = "mps"
# else:
#     DEVICE = "cpu"

# print(f"\nUsing device: {DEVICE}")


# objects to look for

CORRIDOR_CLASSES = [

    # shoes / storage
    "shoe rack",
    "shoes",
    "cabinet",
    "cardboard box",

    # furniture
    "chair",
    "stool",
    "table",

    # plants
    "potted plant",

    # mobility
    "bicycle",
    "electric scooter",
    "baby stroller",
    "wheelchair",

    # trolleys
    "shopping trolley",

    # laundry
    "clothes drying rack",

    # household
    "trash bin",
    "bucket",
    "ladder",

    # other items
    "parcel",
    "large bag",
    "suitcase",

    # safety
    "fire extinguisher"
]


# different objects need different confidence levels

CLASS_THRESHOLDS = {

    "bicycle": 0.25,
    "electric scooter": 0.25,

    "chair": 0.30,
    "stool": 0.30,
    "table": 0.30,

    "shoe rack": 0.30,
    "shoes": 0.30,
    "cabinet": 0.35,

    "cardboard box": 0.25,

    "potted plant": 0.30,

    "baby stroller": 0.25,
    "wheelchair": 0.25,

    "shopping trolley": 0.25,

    "clothes drying rack": 0.25,

    "trash bin": 0.30,
    "bucket": 0.30,
    "ladder": 0.30,

    "parcel": 0.25,
    "large bag": 0.30,
    "suitcase": 0.25,

    "fire extinguisher": 0.35
}


# if object is not listed above
# use this confidence level

DEFAULT_THRESHOLD = 0.30


# storage

overall_counts = defaultdict(int)

video_reports = []

track_reports = []


# find all videos

if not VIDEO_FOLDER.exists():

    print("\nERROR: videos folder does not exist.")

    print(
        "Create a folder called 'videos' first."
    )

    exit()


videos = sorted([

    file

    for file in VIDEO_FOLDER.iterdir()

    if file.suffix.lower()
    in VIDEO_EXTENSIONS

])


print(
    f"\nFound {len(videos)} videos.\n"
)


if len(videos) == 0:

    print(
        "No videos found inside "
        "the videos folder."
    )

    exit()


# load model once

print("Loading YOLOE...")

model = YOLOE(
    MODEL_NAME
)


# tell YOLO exactly what we want to look for

print(
    "Setting corridor classes..."
)

model.set_classes(
    CORRIDOR_CLASSES
)

print("Ready!\n")


# process each video

for video_number, video_path in enumerate(

    videos,

    start=1

):

    print("\n" + "=" * 70)

    print(
        f"Processing "
        f"{video_number}/{len(videos)}"
    )

    print(
        f"Video: {video_path.name}"
    )

    print("=" * 70)


    # storage for this video

    # each track id stores information
    # about the same object across frames

    # eg:
    #
    # track 5:
    # bicycle
    # seen in 20 frames
    # strong detection in 15 frames

    track_stats = defaultdict(
        lambda: {

            "frames_seen": 0,

            "strong_frames": 0,

            "scores": [],

            "scores_by_class":
                defaultdict(list),

            "class_votes":
                defaultdict(int),

            "strong_class_votes":
                defaultdict(int),

            "first_frame": None,

            "last_frame": None
        }
    )


    # store which strong track ids
    # appeared in each frame

    strong_tracks_by_frame = (
        defaultdict(set)
    )


    # run yolo

    results = model.track(

        source=str(video_path),

        stream=True,

        tracker=TRACKER,

        conf=CONFIDENCE,

        imgsz=IMAGE_SIZE,

        vid_stride=VIDEO_STRIDE,

        max_det=MAX_DETECTIONS,

        device=DEVICE,

        save=True,

        project="annotated_videos",

        name=video_path.stem,

        exist_ok=True,

        verbose=True
    )


    # read results frame by frame

    frame_number = 0


    for result in results:

        frame_number += 1


        if result.boxes is None:

            continue


        if len(result.boxes) == 0:

            continue


        # classes detected in this frame

        class_ids = (

            result.boxes.cls

            .int()

            .cpu()

            .tolist()

        )


        confidences = (

            result.boxes.conf

            .cpu()

            .tolist()

        )


        # track ids

        if result.boxes.id is None:

            continue


        track_ids = (

            result.boxes.id

            .int()

            .cpu()

            .tolist()

        )


        # process every tracked object

        for (
            track_id,
            class_id,
            confidence
        ) in zip(

            track_ids,
            class_ids,
            confidences

        ):


            object_name = (
                model.names[
                    class_id
                ]
            )


            # get confidence threshold
            # for this specific object

            threshold = (
                CLASS_THRESHOLDS.get(
                    object_name,
                    DEFAULT_THRESHOLD
                )
            )


            # get storage for this track

            stats = (
                track_stats[
                    track_id
                ]
            )


            # count how many frames
            # this object has appeared in

            stats[
                "frames_seen"
            ] += 1


            # store confidence

            stats[
                "scores"
            ].append(
                confidence
            )


            stats[
                "scores_by_class"
            ][
                object_name
            ].append(
                confidence
            )


            # vote for what object
            # this track probably is

            stats[
                "class_votes"
            ][
                object_name
            ] += 1


            # save first frame

            if (
                stats[
                    "first_frame"
                ]
                is None
            ):

                stats[
                    "first_frame"
                ] = frame_number


            # update last frame

            stats[
                "last_frame"
            ] = frame_number


            # strong detection

            if (
                confidence
                >= threshold
            ):

                stats[
                    "strong_frames"
                ] += 1


                stats[
                    "strong_class_votes"
                ][
                    object_name
                ] += 1


                # remember this strong
                # track appeared in this frame

                strong_tracks_by_frame[
                    frame_number
                ].add(
                    track_id
                )


    # work out which tracks are real

    confirmed_tracks = {}

    confirmed_by_class = (
        defaultdict(list)
    )


    for (
        track_id,
        stats
    ) in track_stats.items():


        # object must appear across
        # multiple frames

        if (
            stats[
                "frames_seen"
            ]
            < MIN_TRACK_FRAMES
        ):

            continue


        # object must have enough
        # strong detections

        if (
            stats[
                "strong_frames"
            ]
            < MIN_STRONG_FRAMES
        ):

            continue


        # use strong detections
        # to decide what the object is

        if (
            stats[
                "strong_class_votes"
            ]
        ):

            object_name = max(

                stats[
                    "strong_class_votes"
                ],

                key=stats[
                    "strong_class_votes"
                ].get

            )


        else:

            object_name = max(

                stats[
                    "class_votes"
                ],

                key=stats[
                    "class_votes"
                ].get

            )


        # confidence scores only
        # for final chosen class

        scores = (

            stats[
                "scores_by_class"
            ][
                object_name
            ]

        )


        if scores:

            average_confidence = (

                sum(scores)
                / len(scores)

            )

            max_confidence = max(
                scores
            )

        else:

            average_confidence = 0

            max_confidence = 0


        confirmed_tracks[
            track_id
        ] = object_name


        confirmed_by_class[
            object_name
        ].append(
            track_id
        )


        # detailed track report

        track_reports.append({

            "video":
                video_path.name,

            "track_id":
                track_id,

            "object":
                object_name,

            "frames_seen":
                stats[
                    "frames_seen"
                ],

            "strong_frames":
                stats[
                    "strong_frames"
                ],

            "first_frame":
                stats[
                    "first_frame"
                ],

            "last_frame":
                stats[
                    "last_frame"
                ],

            "average_confidence":
                round(
                    average_confidence
                    * 100,
                    1
                ),

            "max_confidence":
                round(
                    max_confidence
                    * 100,
                    1
                )
        })


    # maximum number of each object
    # visible in a single frame

    max_simultaneous = (
        defaultdict(int)
    )


    for (
        frame,
        track_ids
    ) in strong_tracks_by_frame.items():


        # count confirmed objects
        # visible in this frame

        frame_counts = (
            defaultdict(int)
        )


        for track_id in track_ids:


            if (
                track_id
                not in
                confirmed_tracks
            ):

                continue


            object_name = (
                confirmed_tracks[
                    track_id
                ]
            )


            frame_counts[
                object_name
            ] += 1


        # maximum visible simultaneously

        for (
            object_name,
            count
        ) in frame_counts.items():


            max_simultaneous[
                object_name
            ] = max(

                max_simultaneous[
                    object_name
                ],

                count

            )


    # video summary

    print(
        f"\nResults for "
        f"{video_path.name}:\n"
    )


    if not confirmed_by_class:

        print(
            "No confirmed objects detected."
        )


    for object_name in sorted(
        confirmed_by_class
    ):


        track_ids = (
            confirmed_by_class[
                object_name
            ]
        )


        confirmed_count = len(
            track_ids
        )


        # get all confidence scores
        # from confirmed tracks

        confidence_scores = []


        total_frames_seen = 0


        for track_id in track_ids:


            stats = (
                track_stats[
                    track_id
                ]
            )


            total_frames_seen += (
                stats[
                    "frames_seen"
                ]
            )


            confidence_scores.extend(

                stats[
                    "scores_by_class"
                ][
                    object_name
                ]

            )


        if confidence_scores:

            average_confidence = (

                sum(
                    confidence_scores
                )
                / len(
                    confidence_scores
                )

            )

        else:

            average_confidence = 0


        if confirmed_count > 0:

            average_frames_seen = (

                total_frames_seen
                / confirmed_count

            )

        else:

            average_frames_seen = 0


        print(

            f"{object_name:25}"

            f"{confirmed_count:5} confirmed | "

            f"max visible: "
            f"{max_simultaneous[object_name]:3} | "

            f"avg confidence: "
            f"{average_confidence * 100:.1f}%"

        )


        # add to overall total

        overall_counts[
            object_name
        ] += confirmed_count


        # add to CSV report

        video_reports.append({

            "video":
                video_path.name,

            "object":
                object_name,

            "confirmed_tracks":
                confirmed_count,

            "max_simultaneous":
                max_simultaneous[
                    object_name
                ],

            "average_confidence":
                round(
                    average_confidence
                    * 100,
                    1
                ),

            "average_frames_seen":
                round(
                    average_frames_seen,
                    1
                )

        })


# create report by video

with open(

    "report_by_video.csv",

    "w",

    newline="",

    encoding="utf-8"

) as file:


    fieldnames = [

        "video",

        "object",

        "confirmed_tracks",

        "max_simultaneous",

        "average_confidence",

        "average_frames_seen"

    ]


    writer = csv.DictWriter(

        file,

        fieldnames=fieldnames

    )


    writer.writeheader()


    writer.writerows(

        video_reports

    )


# create detailed track report

with open(

    "track_report.csv",

    "w",

    newline="",

    encoding="utf-8"

) as file:


    fieldnames = [

        "video",

        "track_id",

        "object",

        "frames_seen",

        "strong_frames",

        "first_frame",

        "last_frame",

        "average_confidence",

        "max_confidence"

    ]


    writer = csv.DictWriter(

        file,

        fieldnames=fieldnames

    )


    writer.writeheader()


    writer.writerows(

        track_reports

    )


# create overall report

with open(

    "overall_report.csv",

    "w",

    newline="",

    encoding="utf-8"

) as file:


    writer = csv.writer(
        file
    )


    writer.writerow([

        "object",

        "total_confirmed_tracks"

    ])


    for (
        object_name,
        count
    ) in sorted(

        overall_counts.items(),

        key=lambda x: x[1],

        reverse=True

    ):


        writer.writerow([

            object_name,

            count

        ])


# print final summary

print("\n")

print("=" * 70)

print("FINAL REPORT")

print("=" * 70)


if not overall_counts:

    print(
        "No confirmed objects detected."
    )


for (
    object_name,
    count
) in sorted(

    overall_counts.items(),

    key=lambda x: x[1],

    reverse=True

):


    print(

        f"{object_name:25}"

        f"{count}"

    )


print("\nDone!")


print("\nCreated:")

print(
    "• report_by_video.csv"
)

print(
    "• track_report.csv"
)

print(
    "• overall_report.csv"
)

print(
    "• annotated_videos/"
)