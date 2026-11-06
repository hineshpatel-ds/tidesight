"""
Shared settings for the whole team.

Owner: App + integration.
Rule: do not change a value here without posting in the team chat first.
Everyone imports these values instead of hard-coding numbers or paths.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLIPS_DIR = DATA_DIR / "clips"                  # input videos (not committed to git)
SPECIES_LIST = DATA_DIR / "species_list.csv"    # master species list
SPECIES_LIBRARY = DATA_DIR / "species_library"  # data/species_library/<species_id>/*.jpg
MODELS_DIR = ROOT / "models"                    # model weights (not committed to git)
OUTPUTS_DIR = ROOT / "outputs"                  # outputs/<clip_id>/...

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# ---------------------------------------------------------------- output file names
SEGMENTS_FILE = "segments.json"       # written from run_triage()
DETECTIONS_FILE = "detections.jsonl"  # written from run_detection() + identify_species()
MAXN_FILE = "maxn.csv"                # written from compute_maxn()
ENERGY_FILE = "emissions.csv"         # written by CodeCarbon

# ---------------------------------------------------------------- triage (Video + classical CV)
TRIAGE_PAD_S = 1.0          # pad every busy segment by this many seconds on each side
TRIAGE_MERGE_GAP_S = 2.0    # merge two busy segments if the gap between them is shorter

# ---------------------------------------------------------------- detection (ML lead)
DETECT_FPS = 5              # frames per second the detector looks at
DETECT_CONF = 0.25          # minimum detector confidence (low on purpose: recall first)
DETECT_IMGSZ = 1024         # Community Fish Detector expects 1024
DETECTOR_WEIGHTS = MODELS_DIR / "cfd-yolov12x-1.00.pt"
TRACKER = "bytetrack.yaml"

# ---------------------------------------------------------------- species (Species ID)
SPECIES_MODEL = "hf-hub:imageomics/bioclip"
UNKNOWN_THRESHOLD = 0.5     # best species score below this -> species = "unknown"

# ---------------------------------------------------------------- labels
FISH_PLACEHOLDER = "fish"   # species value before the species step runs
UNKNOWN_LABEL = "unknown"   # species value when the model is not confident

# ---------------------------------------------------------------- energy
ENERGY_COUNTRY = "CAN"      # CodeCarbon offline mode needs a country code
