# ImageContainer

## Overview

ImageContainer (`container`) is a Python package that provides a unified interface for image data.
It supports NumPy arrays and PIL images with explicit channel order handling (RGB, BGR, GRAY).

For module details, see [src/container/README.md](src/container/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development, install in editable mode:

```bash
pip install -e .
```

Dependencies are installed automatically.
To install dependencies only:

```bash
pip install -r requirements.txt
```

## Example

After installing the package, import it from any directory:

```python
from PIL import Image

from container import ChannelOrder, ImageContainer

image = Image.open("image.png")

img = ImageContainer.register(
    image=image,
    channel_order=ChannelOrder.RGB,
)

array_bgr = img.to_array(ch_order=ChannelOrder.BGR)
cropped = img.crop((slice(50, 200), slice(100, 300)))
cropped.save("out/cropped.png")
```
