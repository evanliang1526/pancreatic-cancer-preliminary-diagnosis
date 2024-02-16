import os
import dicom2nifti
from glob import glob
output_folder = 'C:\\Users\\junpa\\OneDrive\\Desktop\\Control'
input = "C:\\Users\\junpa\\OneDrive\\Desktop\\Root\*"
i = 0
for filename in glob(input):
    dicom2nifti.dicom_series_to_nifti(filename, os.path.join(output_folder, "pancreas_" + str(i+1) + ".nii.gz"))
    i += 1
