import os
import numpy as np
import pydicom
import nibabel as nib

def load_dicom_folder(dicom_folder):
    slices = []
    for dirName, subdirList, fileList in os.walk(dicom_folder):
        for filename in fileList:
            if ".dcm" in filename.lower():
                dicom_path = os.path.join(dirName, filename)
                dicom_data = pydicom.dcmread(dicom_path)
                slices.append(dicom_data)

    # Sort slices by ImagePositionPatient (z-coordinate)
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Stack the pixel arrays along the third dimension
    image = np.stack([s.pixel_array for s in slices], axis=-1)
    return image, slices[0]


def save_nifti(image, reference_dicom, output_filename):
    # Create an affine transformation matrix
    affine = np.eye(4)
    affine[0, 0] = reference_dicom.PixelSpacing[0]
    affine[1, 1] = reference_dicom.PixelSpacing[1]
    affine[2, 2] = reference_dicom.SliceThickness
    affine[0, 3] = reference_dicom.ImagePositionPatient[0]
    affine[1, 3] = reference_dicom.ImagePositionPatient[1]
    affine[2, 3] = reference_dicom.ImagePositionPatient[2]

    # Create and save NIfTI image
    nifti_img = nib.Nifti1Image(image, affine)
    nib.save(nifti_img, output_filename)


def convert_and_save(dicom_root, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    subjects = sorted(os.listdir(dicom_root))
    for subject_index, subject_name in enumerate(subjects, start=1):
        subject_path = os.path.join(dicom_root, subject_name)
        if os.path.isdir(subject_path):
            subject_id = f"{subject_index:03d}"
            contrasts = sorted(os.listdir(subject_path))
            mapping = []
            for contrast_index, contrast_name in enumerate(contrasts, start=1):
                contrast_path = os.path.join(subject_path, contrast_name)
                if os.path.isdir(contrast_path):
                    contrast_id = f"{contrast_index:04d}"
                    dicom_folder = contrast_path
                    output_filename = os.path.join(output_root, f"CDPG_{subject_id}_{contrast_id}.nii")

                    # Load DICOM folder
                    image, reference_dicom = load_dicom_folder(dicom_folder)

                    # Save as NIfTI file
                    save_nifti(image, reference_dicom, output_filename)

                    # Append mapping info
                    mapping.append(f"{subject_name} ({subject_id}), {contrast_name} ({contrast_id})")

            # Save the mapping file for the subject
            mapping_file = os.path.join(output_root, f"CDPG_{subject_id}_mapping.txt")
            with open(mapping_file, 'w') as f:
                f.write("\n".join(mapping))

def parse_file_tree(directory):
    result = {}
    for entry in os.scandir(directory):
        if entry.is_file():
            result[entry.name] = None
        elif entry.is_dir():
            result[entry.name] = parse_file_tree(entry.path)
    return result
def main():
    dicom_root = '../../../Datasets/MRI/CHDI_Multi_Contrast'  # Root directory containing the DICOM files
    output_root = '../../../Datasets/MRI/CHDI_Multi_Contrast/preprocessed'  # Output directory for NIfTI files and mapping files
    file_tree_dict = parse_file_tree(dicom_root)

    convert_and_save(dicom_root, output_root)


if __name__ == '__main__':
    main()