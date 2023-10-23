import pydicom
import numpy as np


def dicom_to_image(dicom_image: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    """
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dicom_image.pixel_array
    if dicom_image.PixelRepresentation == 1:
        bit_shift = dicom_image.BitsAllocated - dicom_image.BitsStored
        dtype = pixel_array.dtype 

        pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
        #pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(pixel_array, dicom_image)


    if dicom_image.PhotometricInterpretation == "MONOCHROME1":
        assert 1 == 2
        pixel_array = 1 - pixel_array

    # transform to hounsfield units
    intercept = float(dicom_image.RescaleIntercept)
    slope = float(dicom_image.RescaleSlope)

    center = int(dicom_image.WindowCenter)
    width = int(dicom_image.WindowWidth)

    low = center - width / 2
    high = center + width / 2    
    
    pixel_array = pixel_array * slope + intercept
    pixel_array = np.clip(pixel_array, low, high)



    # normalization
    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-6)
    pixel_array = (pixel_array * 255).astype(np.uint8)

    return pixel_array








def get_extract_indexes(total_len, extract_num):
    indexes = np.arange(0, extract_num)
    indexes = indexes * (total_len / extract_num)
    indexes = np.floor(indexes).astype(np.int32)
    return indexes





