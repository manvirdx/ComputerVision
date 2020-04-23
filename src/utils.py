import numpy as np
import cv2
import os
import math
from pathlib import Path
import shutil

def create_directory(name):
	Path(name).mkdir(parents=True, exist_ok=True)

def delete_directory(dir_path):
	shutil.rmtree(dir_path)

def concatenate_images(images):
	"""
	Horizontally concatenates a list of n x m images
	Parameters:
	images: A list of n x m images
	Returns:
	An array representing a horizontal concatenation of the input images
	"""
	horizontal_concat = np.concatenate(tuple(images), axis=1)

	return horizontal_concat

def generate_box_blur(size):
	return np.ones((size, size), np.float32) / (size * size)

def get_template_config(template_file, class_name=True):
	"""
	Gets the level and rotation configuration for a given template file
	Parameters:
	template: The template file name
	class_name: Whether or not to include the tempalte class name
	Returns:
	A tuple containing its level, rotation, and if selected, class name respectively
	"""
	config = template_file.replace('.png', '')
	level_start_index = config.find('level') + len('level')
	level_end_index = config[level_start_index:].find('-') + level_start_index
	level = config[level_start_index:level_end_index]

	rotation = config.find('rotation') + len('rotation')
	rotation_start_index = config.find('rotation') + len('rotation')
	rotation = config[rotation_start_index:]

	return (safe_int_cast(level), safe_int_cast(rotation))

def safe_int_cast(value, default=None):
	try:
		return int(value)
	except(ValueError, TypeError):
		return default	

def get_files(directory, extension='.png', remove_extension=True):
	"""
	Gets a list of files in a given directory
	Parameters:
	directory: The directory to analyse
	extension: The file extension to filter by
	remove_extension: Whether or not to remove file extension from file name
	Returns:
	The list of files names
	"""
	if not os.path.isdir(directory):
		return []

	file_names = [name for name in os.listdir(directory) if name.endswith(extension)]

	if remove_extension:
		file_names = [name.replace(extension, '') for name in file_names]
	
	return file_names

def show_image(image, name='Test'):
	cv2.imshow(name, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def gaussian_kernel(rows, columns, dev=1):
	"""
	Generates a gaussian matrix to be used to blur an image when used as a convolution filter.
	:param rows: the width of the kernel
	:param columns: the height of the kernel
	:param dev: the standard deviation
	:return: the kernel as a numpy.ndarray
	"""
	output_matrix = np.zeros((rows, columns))  # initialise output kernel

	matrix_sum = 0
	r = int((rows - 1) / 2)  # used for the loops to leave out borders and to center kernel loop
	c = int((columns - 1) / 2)  # used for the loops to leave out borders and to center kernel loop

	# loop through each row of image then each column (pixel) of that row
	for i in range(-r, r + 1, 1):
		for j in range(-c, c + 1, 1):
			gaussian_value = (1 / (2 * math.pi * (dev ** 2))) * math.exp(((i ** 2) + (j ** 2)) / (2 * (dev ** 2)))
			output_matrix[i + r, j + c] = gaussian_value
			matrix_sum += gaussian_value

	return output_matrix / matrix_sum

def write_to_file(file_dir, text):
	f = open(file_dir, "a")
	f.write(str(text))
	f.close()