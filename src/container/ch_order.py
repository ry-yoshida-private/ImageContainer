from enum import Enum

class ChannelOrder(Enum):
    """
    Channel order for images.

    Attributes
    ----------
    RGB: RGB channel order.
    BGR: BGR channel order.
    GRAY: Gray channel order.
    """
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"