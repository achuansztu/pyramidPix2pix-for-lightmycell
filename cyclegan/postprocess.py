import os
from PIL import Image
import re
import shutil

cwd = os.getcwd()
output_folder = os.path.join(cwd, 'outputs')
os.makedirs(output_folder, exist_ok=True)
for filename in os.listdir(output_folder):
    file_path = os.path.join(output_folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')


for folder_name in ['light2Tubulin', 'light2Mitochondria', 'light2Nucleus', 'light2Actin']:
    folder_a = os.path.join(cwd, f'results/{folder_name}/test_latest/images')
    folder_b = os.path.join(cwd, f'datasets/{folder_name}/testA')
    print(folder_a)
    for filename in os.listdir(folder_a):
        #print(filename)
        if filename.endswith('_fake.png'):

            corresponding_filename = filename.replace('.ome.tiff_fake.png', '.ome.tiff.tif')
            corresponding_filepath = os.path.join(folder_b, corresponding_filename)

            if os.path.exists(corresponding_filepath):

                with Image.open(corresponding_filepath) as img_b:
                    size_b = img_b.size

                with Image.open(os.path.join(folder_a, filename)) as img_a:
                    img_a_resized = img_a.resize(size_b, Image.LANCZOS)  

                if folder_name == 'light2Tubulin':
                    new_filename = re.sub(r'(image_\d+)_.*\.ome\.tiff_fake\.png', r'\1_Tubulin.ome.tiff', filename)
                elif folder_name == 'light2Mitochondria':
                    new_filename = re.sub(r'(image_\d+)_.*\.ome\.tiff_fake\.png', r'\1_Mitochondria.ome.tiff', filename)
                elif folder_name == 'light2Nucleus':
                    new_filename = re.sub(r'(image_\d+)_.*\.ome\.tiff_fake\.png', r'\1_Nucleus.ome.tiff', filename)
                elif folder_name == 'light2Actin':
                    new_filename = re.sub(r'(image_\d+)_.*\.ome\.tiff_fake\.png', r'\1_Actin.ome.tiff', filename)

                output_path = os.path.join(output_folder, new_filename)
                img_a_resized.save(output_path, format='TIFF')

                print(f'Processed and saved: {new_filename}')
            else:
                print(f'No corresponding file found for: {filename}')