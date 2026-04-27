from __future__ import annotations
import os
import numpy as np
import cv2
from PIL import Image
from ..ch_order import ChannelOrder
from ..format import ImageFormat
from ..container import ImageContainer
from ..binary_image import BinaryImage

class ArrayImageContainer(ImageContainer[np.ndarray]):
    """
    Container class for numpy arrays.
    
    Attributes:
    ----------
    value: np.ndarray
        The numpy array.
    channel_order: ChannelOrder
        The channel order of the image.
    """

    def __post_init__(self) -> None:
        """
        Post initialize the array image container.
        """
        super().__post_init__()
        self.value.setflags(write=False)

    @property
    def format(self) -> ImageFormat:
        """The image format (ARRAY)."""
        return ImageFormat.ARRAY    

    def _validate_image(self) -> None:
        """
        Validate the image.

        Parameters:
        ----------
        image: np.ndarray
            The image to validate.
        channel_order: ChannelOrder
            The channel order of the image.

        Raises
        ------
        ValueError:
            If the image is not a 3D numpy array.
            If the image is not a 2D numpy array.
            If the image has the wrong number of channels.
        """
        if self.channel_order in [ChannelOrder.RGB, ChannelOrder.BGR]:
            if self.value.ndim != 3:
                raise ValueError(f"Image must have 3 dimensions. Got {self.value.ndim}")
            if self.value.shape[2] != 3:
                raise ValueError(f"Image must have 3 channels. Got {self.value.shape[2]}")
        if self.channel_order == ChannelOrder.GRAY:
            if self.value.ndim != 2:
                raise ValueError(f"Image must have 2 dimensions. Got {self.value.ndim}")
        self.value.setflags(write=False)

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Get the shape of the image.
        
        Returns
        -------
        tuple[int, int, int]: The shape(height, width, channels) of the image.
        """
        shape = self.value.shape
        if len(shape) == 2:
            return (shape[0], shape[1], 1)
        elif len(shape) == 3:
            return shape
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    
    @property
    def w(self) -> int:
        """
        Get the width of the image.
        
        Returns
        -------
        int: The width of the image.
        """
        return self.shape[1]
    
    @property
    def h(self) -> int:
        """
        Get the height of the image.
        
        Returns
        -------
        int: The height of the image.
        """
        return self.shape[0]

    @property
    def size(self) -> tuple[int, int]:
        """
        Get the size of the image.
        
        Returns
        -------
        tuple[int, int]: The size(height, width) of the image.
        """
        return self.shape[:2]

    @property
    def ch(self) -> int:
        """
        Get the number of channels of the image.
        
        Returns
        -------
        int: The number of channels of the image.
        """
        match self.channel_order:
            case ChannelOrder.GRAY:
                return 1
            case _:
                return 3
    
    def to_PIL(self) -> Image.Image:
        """
        Get the PIL image.
        
        Returns
        -------
        Image.Image: The PIL image.
        """
        if self.channel_order == ChannelOrder.BGR:
            return Image.fromarray(self.value[..., [2, 1, 0]], mode='RGB')
        elif self.channel_order == ChannelOrder.GRAY:
            return Image.fromarray(self.value, mode='L')
        else:
            return Image.fromarray(self.value, mode='RGB')

    def to_array(
        self, 
        ch_order: ChannelOrder = ChannelOrder.BGR
        ) -> np.ndarray:
        """
        Get the array image.
        
        Returns
        -------
        np.ndarray: The array image.

        Raises
        ------
        ValueError:
            If the channel order is not supported.
        """
        if ch_order in (ChannelOrder.BGR, ChannelOrder.RGB):
            if ch_order == self.channel_order:
                return self.value.copy()
            else:
                return self.value[..., ::-1].copy()
        elif ch_order == ChannelOrder.GRAY:
            if ch_order == self.channel_order:
                return self.value.copy()
            if self.channel_order == ChannelOrder.BGR:
                return cv2.cvtColor(self.value, cv2.COLOR_BGR2GRAY)
            else:
                return cv2.cvtColor(self.value, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unsupported channel order: {ch_order}")

    def to_binary(self, threshold: int | float) -> BinaryImage:
        """
        Convert the image into a binary image using a threshold.

        Values >= `threshold` become 1, otherwise 0.
        If the input has 3 channels, it will be converted to gray first.
        """
        gray = self.to_array(ChannelOrder.GRAY)  # (H, W)
        binary01 = (gray >= threshold).astype(np.uint8)  # 0/1
        return BinaryImage(value=binary01)

    def save(self, save_path: str) -> None:
        """
        Save the image to a path.

        Parameters
        ----------
        save_path: str
            The path to save the image.
        """
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self.to_PIL().save(save_path)

    def crop(
        self, 
        crop_slice: tuple[slice, slice]
        ) -> ArrayImageContainer:
        """
        Crop the image.
        
        Parameters:
        ----------
        crop_slice: tuple[slice, slice]
            The slice to crop the image(y_slice, x_slice).
            example: (slice(100, 200), slice(300, 400))

        Returns:
        ----------
        ArrayImageContainer: The cropped image container.
        """
        return ArrayImageContainer(
            value=self.value[crop_slice],
            channel_order=self.channel_order
            )

    def _to_array(self) -> np.ndarray:
        """
        Convert the image to a numpy array. 
        
        Returns
        -------
        np.ndarray: The image as a numpy array.
        """
        return self.value

    def to_ch_swapped_image(
        self, 
        output_order: ChannelOrder = ChannelOrder.BGR
        ) -> np.ndarray:
        """
        Get the channel swapped image.
        
        Parameters:
        ----------
        output_order: ChannelOrder
            The output channel order to get the image.

        Returns
        -------
        np.ndarray: The channel swapped image.
        """
        if self.channel_order == output_order:
            return self.value
        array = self._create_swapped_array(output_order)
        return array

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, w={self.w}, h={self.h}, channel_order={self.channel_order})"

    @classmethod
    def from_path(
        cls, 
        image_path: str, 
        ) -> ArrayImageContainer:
        """
        Create an array image container from an image path.
        
        Parameters:
        ----------
        image_path: str
            The path to the image.
        
        Returns:
        ----------
        ArrayImageContainer: The array image container.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        channel_order = ChannelOrder.BGR
        return cls(
            value=image, 
            channel_order=channel_order
            )