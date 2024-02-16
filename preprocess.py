import os
import numpy as np
import nibabel as nib
import skimage.transform as skTrans

# Specify the directories
input_cancer = "C:\\Users\\junpa\\OneDrive\\Desktop\\Cancer"
output_cancer = "C:\\Users\\junpa\\OneDrive\\Desktop\\Processed_Cancer"
input_control = "C:\\Users\\junpa\\OneDrive\\Desktop\\Control"
output_control = "C:\\Users\\junpa\\OneDrive\\Desktop\\Processed_Control"

# Process and save Cancer images
for filename in os.listdir(input_cancer):
    full_path = os.path.join(input_cancer, filename)
    im = nib.load(full_path).get_fdata()
    new_shape = (400, 400, 15, 1)
    resized_volume = skTrans.resize(im, new_shape)
    grayscale_volume = np.mean(resized_volume, axis=-1, keepdims=True)
    output_filename = os.path.splitext(filename)[0] + '_processed.nii'
    output_path = os.path.join(output_cancer, output_filename)
    nib.save(nib.Nifti1Image(grayscale_volume, np.eye(4)), output_path)

# Process and save Control images
for filename in os.listdir(input_control):
    full_path = os.path.join(input_control, filename)
    im = nib.load(full_path).get_fdata()
    new_shape = (400, 400, 15, 1)
    resized_volume = skTrans.resize(im, new_shape)
    grayscale_volume = np.mean(resized_volume, axis=-1, keepdims=True)
    output_filename = os.path.splitext(filename)[0] + '_processed.nii'
    output_path = os.path.join(output_control, output_filename)
    nib.save(nib.Nifti1Image(grayscale_volume, np.eye(4)), output_path)
