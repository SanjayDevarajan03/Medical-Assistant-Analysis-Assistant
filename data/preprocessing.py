import pandas as pd
import torch
from PIL import Image
import pydicom
from pathlib import Path
from config import Config
from torchvision import transforms
from torch.utils.data import Dataset

def read_dicom(dicom_path):
    """
    Read xray file and convert to numpy array

    Args:
        xray_path: Path to xray file

    Returns:
        image_array: Numpy array of pixel data
        metadata: Dict containign relevatn xray metadata
    """
    try:
        dicom = pydicom.dcmread(dicom_path)

        # Extract relevant metadata
        metadata = {
            'window_center': dicom.get('WindowCenter', Config.WINDOW_CENTER),
            'window_width': dicom.get('WindowWidth', Config.WINDOW_WIDTH),
            'modality': dicom.get('Modality', None),
            'patient_id': dicom.get('PatientID','unknown'),
            'study_description': dicom.get('StudyDescription', None),
            'image_description': dicom.get('ImageOrientationPatient', None),
        }
        # Handle window center/width as signle value or multiple values
        if hasattr(metadata['window_center'], '__iter__') and not isinstance(metadata[""]):
            pass

    except:
        pass



def apply_windowing(image, window_center, window_width):
    """
    Apply windowing to adjust contrast of medical images

    Args:
        image: Input image as numpy array
        window_center: Center of window in HU(Hound)
    """

