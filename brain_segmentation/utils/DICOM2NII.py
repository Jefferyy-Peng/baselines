import os
import SimpleITK as sitk

def dicom_to_nifti(dicom_dir, output_filename):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    sitk.WriteImage(image, output_filename)
    print(f"✔ Saved: {output_filename}")

# Set paths
dataset_root = "/home/yxpengcs/Datasets/MRI/CHDI_Multi_Contrast/SyMRI_processed_DL"
targets = ["T1W", "T2W", "PSIR"]

# Iterate over patient directories
for patient in os.listdir(dataset_root):
    patient_path = os.path.join(dataset_root, patient)
    if not os.path.isdir(patient_path):
        continue  # Skip non-directory entries

    for folder in os.listdir(patient_path):
        for target in targets:
            if target in folder:  # Match e.g., "1200_T1W_AX"
                dicom_path = os.path.join(patient_path, folder)
                output_filename = os.path.join(patient_path, f"{target}.nii.gz")

                try:
                    dicom_to_nifti(dicom_path, output_filename)
                except Exception as e:
                    print(f"Failed to convert {dicom_path}: {e}")