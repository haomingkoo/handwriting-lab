"""API and page tests for the inference service."""

import io

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from mnist_fastapi.main import APP

CLIENT = TestClient(APP)


def _digit_png() -> bytes:
    """A white-on-black vertical stroke, roughly an MNIST-style '1'."""
    image = Image.new("L", (28, 28), 0)
    ImageDraw.Draw(image).line((14, 4, 14, 24), fill=255, width=3)
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def test_predict_returns_probability_per_digit():
    response = CLIENT.post(
        "/api/v1/model/predict",
        files={"image_file": ("drawing.png", _digit_png(), "image/png")},
    )

    assert response.status_code == 200
    result = response.json()["data"][0]
    probs = result["probabilities"]
    assert len(probs) == 10
    assert abs(sum(probs) - 1) < 0.01
    assert int(result["prediction"]) == probs.index(max(probs))
    assert abs(result["confidence"] - max(probs)) < 0.001


def test_predict_rejects_non_image():
    response = CLIENT.post(
        "/api/v1/model/predict",
        files={"image_file": ("note.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_index_has_shared_footer_and_no_internal_labels():
    html = CLIENT.get("/").text

    assert "&copy; 2026 Haoming Koo" in html
    assert "privacy@kooexperience.com" in html
    assert "/favicon.svg" in html
    # Internal plumbing stays out of the page copy.
    for leaked in ("API:", "fastapi + canvas", "Latency", "drawing.png</"):
        assert leaked not in html
    assert "—" not in html


def test_static_pages_served():
    for path in ("/privacy", "/terms", "/site.css", "/favicon.svg", "/fonts/fonts.css", "/reports/evaluation_latest.json"):
        assert CLIENT.get(path).status_code == 200, path
