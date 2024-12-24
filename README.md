# Parking Sign Interpretation
**Aim:** Take an image input of a parking sign and interpret at what times you can park

## PIPELINE: 
### 1. Pre-process image
- Convert to grayscale
- Use adaptive thresholding/edge detection
- Remove noise

Optional:
- Complex Layouts: For signs with multiple sections, segment the image into regions using OpenCV's contour detection or template matching.
- Low-quality Images: Use image enhancement techniques like histogram equalization or super-resolution.
### 2. Perform OCR 
### 3. Parse text using regex


## Final Output of Sign:
JSON file with following structure:
```
{"parking_times": 
    [
        {"start_time": "09:00",
        "end_time": "17:30,
        "days": ["Mon", "Tues", "Wed", "Thurs", "Fri"],
        "ticketed": False,
        "permit": null,
        "max_duration": 0.25
        },
        {"start_time": "09:00",
        "end_time": "12:00,
        "days": ["Sat"],
        "ticketed": False,
        "permit": null,
        "max_duration": 0.25
        }
    ]
}
```