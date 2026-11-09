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
PROTOCOL_FILE = "protocol.json"
COUNTS_FILE = "counts.json"
SAMPLE_COUNTS_FILE = "sample_counts.csv"

# ---------------------------------------------------------------- triage (Video + classical CV)
TRIAGE_PAD_S = 1.0          # pad every busy segment by this many seconds on each side
TRIAGE_MERGE_GAP_S = 2.0    # merge two busy segments if the gap between them is shorter

# ---------------------------------------------------------------- detection (ML lead)
DETECT_FPS = 5              # frames per second the detector looks at
DETECTOR_WEIGHTS = MODELS_DIR / "tidesight-fish-v1.pt"
DETECT_CONF = 0.18
DETECT_IMGSZ = 1280
DETECT_INPUT_WIDTH = 1920
DETECT_PREPROCESS = "stretch1-99_clahe3-8x8_v1"
TRACK_LINK_DIST = 1.5
TRACK_MAX_MISSED = 2
TRACK_MAX_GAP_S = 1.0
# Protocol sampling defaults.
PROTOCOL_N_SAMPLES = 15
PROTOCOL_INTERVAL_S = 60.0
PROTOCOL_EDGE_S = 60.0      # discard this duration at both ends of the source video

# ---------------------------------------------------------------- species (Species ID)
SPECIES_MODEL = "hf-hub:imageomics/bioclip"
UNKNOWN_THRESHOLD = 0.5     # best species score below this -> species = "unknown"

# ---------------------------------------------------------------- labels
FISH_PLACEHOLDER = "fish"   # species value before the species step runs
UNKNOWN_LABEL = "unknown"   # species value when the model is not confident

# ---------------------------------------------------------------- energy
ENERGY_COUNTRY = "CAN"      # CodeCarbon offline mode needs a country code
ENERGY_REGION = "nova scotia"
