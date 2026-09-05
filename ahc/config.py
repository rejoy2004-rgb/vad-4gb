"""Shared configuration: paths, label set, prompt bank."""
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRAIN = DATA / "train"
TEST = DATA / "test"
# Overridable so a different encoder can be evaluated end to end without
# disturbing the cache and heads of a working configuration.
CACHE = ROOT / os.environ.get("AHC_CACHE", "cache")
RUNS = ROOT / os.environ.get("AHC_RUNS", "runs")

for _d in (CACHE, RUNS):
    _d.mkdir(parents=True, exist_ok=True)

# Index 0 is the negative class. The submission format forbids emitting
# "normal" as an event, so it is only ever used internally.
CLASSES = [
    "normal",
    "traffic_accident",
    "traffic_congestion",
    "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic",
    "wrong_way_driving",
    "road_spill_or_debris",
    "waterlogging_or_flood",
    "fire",
    "smoke",
    "fighting_or_violence",
    "loitering_or_suspicious_presence",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
ANOMALY_CLASSES = CLASSES[1:]

# Frame sampling. Anomalies like an accident last ~1s, so we cannot go much
# below 2 fps without stepping over them entirely.
FPS = 2.0
MAX_FRAMES = 2048

ENCODER_ID = os.environ.get("AHC_ENCODER", "google/siglip2-base-patch16-224")

# Zero-shot text prompts, used as a prior and as a cold-start fallback before
# the head is trained. Several phrasings per class, averaged in embedding space.
PROMPTS = {
    "normal": [
        "an ordinary street scene with traffic flowing normally",
        "a routine aerial view of a road with nothing unusual happening",
        "an empty road, a park, or a parking lot with nothing remarkable",
        "people walking normally on a sidewalk",
    ],
    "traffic_accident": [
        "a traffic accident, two vehicles colliding on the road",
        "a crashed car with visible damage after a collision",
        "an overturned vehicle on the roadway after a crash",
        "the aftermath of a road accident with emergency responders",
    ],
    "traffic_congestion": [
        "heavy traffic congestion, a long queue of stopped vehicles",
        "a traffic jam with bumper to bumper cars filling every lane",
        "gridlocked traffic seen from above",
    ],
    "stalled_or_broken_down_vehicle": [
        "a broken down vehicle stopped on the highway shoulder",
        "a stalled car parked on the side of a fast road with hazard lights",
        "a disabled vehicle stopped where vehicles should not stop",
    ],
    "vehicle_blocking_traffic": [
        "a vehicle illegally stopped in a live traffic lane blocking other cars",
        "a truck parked across the road obstructing traffic flow",
        "a vehicle blocking an intersection",
    ],
    "wrong_way_driving": [
        "a vehicle driving the wrong way against oncoming traffic",
        "a car travelling in the wrong direction on a one way road",
        "a vehicle going against the flow of traffic",
    ],
    "road_spill_or_debris": [
        "debris scattered across the road surface",
        "a spill of cargo or material covering the roadway",
        "obstacles and rubble blocking the drivable area of a road",
    ],
    "waterlogging_or_flood": [
        "a flooded street submerged under water",
        "waterlogging covering the road after heavy rain",
        "an aerial view of flooded houses and fields under water",
    ],
    "fire": [
        "a building on fire with visible flames",
        "an intense fire burning with orange flames",
        "a vehicle engulfed in flames",
    ],
    "smoke": [
        "thick smoke rising over the scene",
        "a large plume of grey smoke drifting across the sky",
        "smoke billowing from the ground without visible flames",
    ],
    "fighting_or_violence": [
        "people fighting violently in public",
        "a physical altercation between several people",
        "an assault captured on surveillance camera",
    ],
    "loitering_or_suspicious_presence": [
        "a person loitering suspiciously in one place for a long time",
        "a person standing next to an abandoned bag or suitcase",
        "someone lingering in an area where nobody normally stays",
    ],
}
