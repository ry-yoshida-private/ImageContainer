from __future__ import annotations
import os
import numpy as np
from PIL import Image

from ..ch_order import ChannelOrder
from ..format import ImageFormat
from ..container import ImageContainer
from ..binary_image import BinaryImage

class PILImageContainer(ImageContainer[Image.Image]):
    """
    Container class for PIL images.
    
    Attributes:
    ----------
    value: Image.Image
        The PIL image.
    channel_order: ChannelOrder
        The channel order of the image.
    """

    def _validate_image(self) -> None:
        """
        Validate the image.
        """
        pass

    @property
    def format(self) -> ImageFormat:
        """The image format (PIL)."""
        return ImageFormat.PIL

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Get the shape of the image.
        
        Returns
        -------
        tuple[int, int, int]: The shape(height, width, channels) of the image.
        """
        return (self.h, self.w, self.ch)
    
    @property
    def w(self) -> int:
        """
        Get the width of the image.
        
        Returns
        -------
        int: The width of the image.
        """
        return self.value.width
    
    @property
    def h(self) -> int:
        """
        Get the height of the image.
        
        Returns
        -------
        int: The height of the image.
        """
        return self.value.height
    
    @property
    def size(self) -> tuple[int, int]:
        """
        Get the size of the image.
        
        Returns
        -------
        tuple[int, int]: The size(width, height) of the image.
        """
        return self.value.size

    @property
    def ch(self) -> int:
        """
        Get the number of channels of the image.
        
        Returns
        -------
        int: The number of channels of the image.
        """
        mode = self.value.mode
        match mode:
            case 'L':
                return 1
            case 'RGB':
                return 3
            case 'RGBA':
                return 4
            case _:
                raise ValueError(f"Unsupported PIL mode: {self.value.mode}")

    def to_PIL(self) -> Image.Image:
        """
        Get the PIL image.
        
        Returns
        -------
        Image.Image: The PIL image.
        """
        return self.value

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
        match ch_order:
            case ChannelOrder.RGB:
                return np.array(self.value)
            case ChannelOrder.BGR:
                return np.array(self.value)[..., [2, 1, 0]]
            case ChannelOrder.GRAY:
                return np.array(self.value.convert('L'))

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
        self.value.save(save_path)
     
    def crop(
        self, 
        crop_slice: tuple[slice, slice]
        ) -> PILImageContainer:
        """
        Crop the image.
        
        Parameters:
        ----------
        crop_slice: tuple[slice, slice]
            The slice to crop the image(y_slice, x_slice).
            example: [100:200, 300:400]

        Returns:
        ----------
        PILImageContainer: The cropped image container.
        """
        y_slice, x_slice = crop_slice
        left = x_slice.start or 0
        upper = y_slice.start or 0
        right = x_slice.stop or self.value.width
        lower = y_slice.stop or self.value.height

        cropped_image = self.value.crop((left, upper, right, lower))
        return PILImageContainer(
            value=cropped_image, 
            channel_order=self.channel_order
            )

    def _to_array(self) -> np.ndarray:
        """
        Convert the image to a numpy array.
        
        Returns
        -------
        np.ndarray: The image as a numpy array.
        """
        array = np.array(self.value)
        
        if len(array.shape) == 2:
            array = np.expand_dims(array, axis=2)
        
        if array.shape[2] == 4:
            array = array[:, :, :3]
        
        return array

    def to_ch_swapped_image(
        self, 
        output_order: ChannelOrder
        ) -> Image.Image:
        """
        Get the channel swapped image.
        
        Parameters:
        ----------
        output_order: ChannelOrder
            The output channel order to get the image.
        """
        if self.channel_order == output_order:
            return self.value

        array = self._create_swapped_array(output_order)

        match output_order:
            case ChannelOrder.RGB | ChannelOrder.BGR:
                return Image.fromarray(array, mode='RGB')
            case ChannelOrder.GRAY:
                return Image.fromarray(array, mode='L')

    @classmethod
    def from_path(
        cls, 
        image_path: str, 
        ) -> PILImageContainer:
        """
        Create a PIL image container from an image path.
        
        Parameters:
        ----------
        image_path: str
            The path to the image.
        
        Returns:
        ----------
        PILImageContainer: The PIL image container.
        """
        image = Image.open(image_path)
        channel_order = ChannelOrder.RGB
        return cls(
            value=image, 
            channel_order=channel_order
            )
                
                



 