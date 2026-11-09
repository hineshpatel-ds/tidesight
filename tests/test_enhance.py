"""Enhancement tests (Video + classical CV). Run with:  python -m pytest -q tests/test_enhance.py"""
import cv2
import numpy as np
import pytest

from tidesight.enhance import colour_params, correct_colour, enhance

# ------------------------------------------------------------------ a scene seen through green water
W, H = 320, 180
GREYS = (40, 90, 140, 190, 240)


def scene():
    """Sand with a row of grey patches, a red patch and an orange fish. Returns (BGR float 0-1, boxes)."""
    img = np.full((H, W, 3), (150, 180, 205), np.float32)
    img += np.random.default_rng(0).normal(0, 12, (H, W, 1)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 2)
    boxes = {}
    for i, v in enumerate(GREYS):
        x = 20 + i * 35
        img[30:65, x:x + 30] = v
        boxes[v] = (x, 30, x + 30, 65)
    img[100:135, 20:55] = (40, 40, 200)
    boxes["red"] = (20, 100, 55, 135)
    cv2.ellipse(img, (260, 130), (35, 14), 0, 0, 360, (40, 120, 230), -1)
    boxes["fish"] = (240, 125, 280, 135)
    return np.clip(img, 0, 255) / 255, boxes


def through_water(true_bgr, beta=(0.35, 0.12, 0.70), backscatter=(0.32, 0.45, 0.12)):
    """Simple underwater model: light fades with distance (red fastest) and green haze is added."""
    distance = np.linspace(4.0, 1.5, H, dtype=np.float32)[:, None, None]
    t = np.exp(-np.array(beta, np.float32) * distance)
    seen = true_bgr * t + np.array(backscatter, np.float32) * (1 - t)
    seen = seen * 255 + np.random.default_rng(1).normal(0, 3, seen.shape)
    return np.clip(seen, 0, 255).astype(np.uint8)


def lab_mean(img, box):
    x1, y1, x2, y2 = box
    return cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float).mean(0)


def grey_cast(img, boxes):
    """Average colourfulness of the grey patches (0 = perfectly grey)."""
    return np.mean([np.hypot(a - 128, b - 128) for _, a, b in (lab_mean(img, boxes[v]) for v in GREYS)])


def grey_spread(img, boxes):
    """Brightness difference between the darkest and the lightest grey patch (contrast)."""
    return lab_mean(img, boxes[GREYS[-1]])[0] - lab_mean(img, boxes[GREYS[0]])[0]


WATERS = {
    "green": {},
    "very murky": dict(beta=(0.5, 0.2, 0.9), backscatter=(0.30, 0.50, 0.15)),
    "bluish": dict(beta=(0.08, 0.10, 0.55), backscatter=(0.45, 0.35, 0.08)),
}


@pytest.fixture(params=WATERS.values(), ids=WATERS.keys())
def murky(request):
    truth, boxes = scene()
    return through_water(truth, **request.param), boxes


# ------------------------------------------------------------------ contract
@pytest.mark.parametrize("shape", [(360, 640, 3), (1080, 1920, 3), (7, 3, 3), (1, 1, 3)])
def test_same_shape_and_dtype(shape):
    frame = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    out = enhance(frame)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_input_is_not_changed():
    frame, _ = scene()
    frame = through_water(frame)
    before = frame.copy()
    enhance(frame)
    assert np.array_equal(frame, before)


@pytest.mark.parametrize("bad", [None, np.zeros((10, 10), np.uint8), np.zeros((10, 10, 4), np.uint8),
                                 np.zeros((10, 10, 3), np.float32), np.zeros((0, 10, 3), np.uint8)])
def test_rejects_what_is_not_a_bgr_frame(bad):
    with pytest.raises(ValueError):
        enhance(bad)


@pytest.mark.parametrize("bgr", [(0, 0, 0), (255, 255, 255), (128, 128, 128), (0, 255, 0), (90, 140, 60)])
def test_plain_frames_do_not_break(bgr):
    """Blank frames (lens cap, open water) must not crash or turn black."""
    frame = np.tile(np.uint8(bgr), (40, 60, 1))
    out = enhance(frame)
    assert out.shape == frame.shape
    if max(bgr) > 0:
        assert out.mean() > 30


def test_black_stays_dark():
    assert enhance(np.zeros((40, 60, 3), np.uint8)).mean() < 30


# ------------------------------------------------------------------ does it actually help
def test_green_tint_is_removed(murky):
    raw, boxes = murky
    assert grey_cast(enhance(raw), boxes) < 0.75 * grey_cast(raw, boxes)


def test_contrast_is_restored(murky):
    raw, boxes = murky
    assert grey_spread(enhance(raw), boxes) > 1.3 * grey_spread(raw, boxes)


def test_red_and_orange_come_back(murky):
    """Red and orange (a* > 0 in LAB) turn grey/green underwater; enhancement brings them back."""
    raw, boxes = murky
    out = enhance(raw)
    for name in ("red", "fish"):
        assert lab_mean(out, boxes[name])[1] - 128 > 10
        assert lab_mean(out, boxes[name])[1] > lab_mean(raw, boxes[name])[1] + 20


def flat_water_frame():
    """Flat murky water with faint 8x8 video-compression blocks (+-1 level), specks and a dark fish."""
    rng = np.random.default_rng(0)
    blocks = rng.integers(-1, 2, (H // 8, W // 8, 1)).repeat(8, 0).repeat(8, 1)
    frame = (np.array([70, 100, 62]) + blocks).astype(np.float32)
    for x, y in rng.uniform([0, 0], [W, H], (60, 2)).astype(int):
        cv2.circle(frame, (int(x), int(y)), 1, (110, 130, 105), -1)
    cv2.ellipse(frame, (160, 90), (40, 12), 0, 0, 360, (30, 40, 35), -1)
    return np.clip(frame, 0, 255).astype(np.uint8)


def fish_clarity(img):
    """Fish-vs-water brightness difference divided by blotchiness of the open water (specks left out)."""
    l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    blotches = (l - cv2.GaussianBlur(l, (0, 0), 4))[10:60, 10:310]
    water = l[10:60, 10:310]
    not_speck = np.abs(water - np.median(water)) < 15
    return abs(np.median(water) - np.median(l[86:94, 140:180])) / blotches[not_speck].std()


def test_flat_water_does_not_turn_blotchy():
    """Stretch + CLAHE also boost compression blocks. With CLAHE 2.0 and a small denoise the
    fish stood out only half as clearly as in the raw frame (0.51); the current settings give 0.74."""
    frame = flat_water_frame()
    assert fish_clarity(enhance(frame)) > 0.65 * fish_clarity(frame)


def test_colour_params_can_be_reused_across_frames():
    """For a steady demo video: measure once, apply the same correction to every frame."""
    truth, _ = scene()
    raw = through_water(truth)
    params = colour_params(raw)
    assert np.array_equal(correct_colour(raw, params), correct_colour(raw))
    assert all(g <= 9.0 + 1e-6 for g in params.gain)   # white balance and stretch are each capped at 3
