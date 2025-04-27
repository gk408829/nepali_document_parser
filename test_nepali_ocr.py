import pytest
import cv2
import numpy as np
from PIL import Image
import tempfile
import os  # Import the os module

from nepali_ocr import (
    remove_small_artifacts,
    preprocess_image,
    deskew,
    clean_text,
    parse_image,
    parse_pdf,
)  # Import your functions

# --- Test Image Processing Functions ---
def create_sample_image():
    """Creates a simple black and white image for testing."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[20:80, 20:80] = 255  # White square in the center
    return image

def test_remove_small_artifacts():
    image = create_sample_image()
    # Add a small artifact (a single white pixel)
    image[5, 5] = 255
    cleaned_image = remove_small_artifacts(image, min_area=2)
    # Check that the artifact is removed
    assert cleaned_image[5, 5] == 0
    # Check that the rest of the image is unchanged
    assert np.array_equal(cleaned_image[20:80, 20:80], image[20:80, 20:80])

def test_preprocess_image():
    image = create_sample_image()
    processed_image = preprocess_image(image)
    # Add more specific assertions based on what preprocess_image should do
    assert processed_image.shape == (100, 100)
    assert processed_image.dtype == np.uint8
    # Further tests: check if denoising affects a noisy image
    noisy_image = image.copy()
    noisy_image[::5, ::5] = 100  # Add salt-and-pepper noise
    processed_noisy_image = preprocess_image(noisy_image, denoise_strength=10)
    assert not np.array_equal(processed_noisy_image, noisy_image)

def test_deskew():
    image = create_sample_image()
    # Create a slightly rotated image for testing
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 5, 1.0)  # Rotate by 5 degrees
    rotated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    deskewed_image = deskew(rotated_image)
    # Check that the deskewed image is similar to the original
    assert deskewed_image.shape == (100, 100)
    assert deskewed_image.dtype == np.uint8

# --- Test Text Cleaning Function ---
def test_clean_text():
    text1 = "This is a test.  with some  extra   spaces and some non-Nepali chars 123 !@#$ कखग"
    cleaning_options1 = {
        "remove_non_nepali": True,
        "remove_whitespace": True,
        "remove_special_chars": True,
        "fix_common_errors": False,
    }
    cleaned_text1 = clean_text(text1, cleaning_options1)
    assert cleaned_text1 == ". - 123 कखग"

    text2 = "oO0123456789"
    cleaning_options2 = {
        "remove_non_nepali": False,
        "remove_whitespace": False,
        "remove_special_chars": False,
        "fix_common_errors": True,
    }
    cleaned_text2 = clean_text(text2, cleaning_options2)
    assert cleaned_text2 == "ोओ०१२३४५६७८९"

    text3 = "abcde"
    cleaning_options3 = {
        "remove_non_nepali": False,
        "remove_whitespace": False,
        "remove_special_chars": False,
        "fix_common_errors": False,
        "custom_replacements": {"a": "b", "c": "d"},
    }
    cleaned_text3 = clean_text(text3, cleaning_options3)
    assert cleaned_text3 == "bbdde"

# --- Test Parsing Functions ---
def test_parse_image():
    # Create a dummy image file
    image = create_sample_image()
    image_file = Image.fromarray(image)
    processing_params = {
        "use_preprocessing": False,
        "denoise_strength": 10,
        "min_smudge_size": 15,
        "use_deskewing": False,
    }
    cleaning_options = {
        "remove_non_nepali": True,
        "remove_whitespace": True,
        "remove_special_chars": True,
        "fix_common_errors": True,
    }

    text, processed_image = parse_image(image_file, processing_params, cleaning_options)
    assert isinstance(text, str)
    assert processed_image is None

def test_parse_pdf():
    # Create a dummy PDF file
    image = create_sample_image()
    image_file = Image.fromarray(image)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        image_file.save(temp_pdf.name)
        pdf_path = temp_pdf.name

    processing_params = {
        "use_preprocessing": False,
        "denoise_strength": 10,
        "min_smudge_size": 15,
        "use_deskewing": False,
    }
    cleaning_options = {
        "remove_non_nepali": True,
        "remove_whitespace": True,
        "remove_special_chars": True,
        "fix_common_errors": True,
    }
    text, processed_images = parse_pdf(pdf_path, processing_params, cleaning_options)
    assert isinstance(text, str)
    assert isinstance(processed_images, list)
    assert all(isinstance(img, Image.Image) for img in processed_images)
    os.unlink(pdf_path)
