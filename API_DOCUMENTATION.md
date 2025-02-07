# BetaCensor2 API Documentation

## Overview
BetaCensor2 is a powerful image censoring API that provides various effects and detection capabilities for sensitive content in images. The API offers multiple censoring effects and can detect various body parts and regions that might need censoring.

## Base URL
The API is accessible at:
- Development: `http://localhost:5000`

## API Endpoints

### 1. Health Check
Check if the API is running properly.

**Endpoint:** `GET /`

**Response:**
```json
{
    "status": "healthy",
    "message": "API is running"
}
```

### 2. Process Image
Process an image with various censoring effects.

**Endpoint:** `POST /process`

**Request Body:**
```json
{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "effect": "pixelation",
    "enabled_parts": ["face", "eyes"],
    "strength": 7,
    "sticker_category": "emoji"
}
```

**Parameters:**
- `image` (required): Base64 encoded image data with data URI scheme
- `effect` (optional): Type of censoring effect to apply
  - Options: `pixelation`, `blur`, `blackbox`, `ruin`, `sticker`
  - Default: `pixelation`
- `enabled_parts` (optional): List of body parts to detect and censor
  - Available options:
    - `face`: Full face detection
    - `eyes`: Eye region detection
    - `mouth`: Mouth region detection
    - `exposed_breast_f`: Exposed female breast
    - `covered_breast_f`: Covered female breast
    - `exposed_genitalia_f`: Exposed female genitalia
    - `covered_genitalia_f`: Covered female genitalia
    - `exposed_breast_m`: Exposed male breast
    - `exposed_genitalia_m`: Exposed male genitalia
    - `exposed_buttocks`: Exposed buttocks
    - `covered_buttocks`: Covered buttocks
    - `belly`: Belly region
    - `feet`: Feet detection
- `strength` (optional): Strength of the censoring effect
  - Range: 1-10
  - Default: 7
- `sticker_category` (optional): Category of stickers to use (only for sticker effect)

**Response:**
```json
{
    "processed_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "regions": [
        {
            "type": "face",
            "coords": [100, 100, 200, 200],
            "confidence": 0.95,
            "label": "face",
            "detection_type": "dlib"
        }
    ]
}
```

**Response Fields:**
- `processed_image`: Base64 encoded processed image data
- `regions`: Array of detected and processed regions
  - `type`: Type of detected region
  - `coords`: Array of coordinates [x, y, width, height]
  - `confidence`: Detection confidence score (0-1)
  - `label`: Detailed label of the detected region
  - `detection_type`: Method used for detection

### Error Responses

The API uses standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `500`: Server Error

Error Response Format:
```json
{
    "error": "Detailed error message"
}
```

## Effects Description

1. **Pixelation**
   - Creates a mosaic-like effect
   - Strength controls pixel size

2. **Blur**
   - Applies Gaussian blur
   - Strength controls blur radius

3. **Blackbox**
   - Covers region with solid black rectangle
   - Strength parameter is ignored

4. **Ruin**
   - Applies artistic distortion effect
   - Includes RGB shift, noise, and contrast adjustments
   - Strength parameter is ignored

5. **Sticker**
   - Overlays stickers on detected regions
   - Requires sticker_category parameter
   - Strength parameter is ignored

## Best Practices

1. **Image Format**
   - Send images in JPEG or PNG format
   - Include the data URI scheme in base64 encoded images
   - Recommended maximum image size: 4096x4096 pixels

2. **Performance**
   - Enable only the necessary body parts for detection
   - Use appropriate strength values (higher values may impact performance)
   - Consider image size and quality trade-offs

3. **Error Handling**
   - Always handle potential error responses
   - Implement proper timeout handling (recommended: 30 seconds)
   - Validate image format and size before sending

## Example Usage

```python
import requests
import base64

# Read and encode image
with open("image.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Add data URI scheme
image_data = f"data:image/jpeg;base64,{encoded_string}"

# Prepare request
payload = {
    "image": image_data,
    "effect": "pixelation",
    "enabled_parts": ["face", "eyes"],
    "strength": 7
}

# Send request
response = requests.post("http://localhost:5000/process", json=payload)

# Handle response
if response.status_code == 200:
    result = response.json()
    processed_image = result["processed_image"]
    detected_regions = result["regions"]
else:
    error = response.json()["error"]
    print(f"Error: {error}")
```

## Rate Limiting and Security

- CORS is enabled for specific origins:
  - `http://localhost:5173`
  - `http://127.0.0.1:5173`
  - `http://localhost:3000`
- Supports credentials for authenticated requests
- Implements standard security headers
- Logs all requests for monitoring and debugging

## Support

For additional support or to report issues, please refer to the project's GitHub repository or contact the development team. 