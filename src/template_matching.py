import cv2
import numpy as np
import math
from convolutions import built_in_convolution

def get_class_name(file_name):
	"""
	Gets the class name for a given file
	Parameters:
	file_name: The file name to analyse
	Returns:
	A string containing the class name
	"""

	return file_name[4:]

def pre_process_image(image):
	# Replace transparent background with white
	trans_mask = image[:,:,3] == 0
	image[trans_mask] = [255, 255, 255, 255]
	filtered_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

	return replace_pixels(filtered_img)

def replace_pixels(image, colour_to_replace=255, with_colour=0):
	"""
	Replaces all pixels of specified colour in the image with ones of another colour.
	Parameters:
	image: An n x m x 3 numpy array representing the image
	colour_to_replace: The rgb colour to replace
	with_colour: The rgb colour to fill in
	Returns:
	An n x m x 3 numpy array representing the new image
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

	image[thresh == 255] = 0

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	erosion = cv2.erode(image, kernel, iterations = 1)
	return erosion

def normalize_image(img):
	"""
	Normalizes an image.
	:param img: the image to normalize
	:return: the normalized image
	"""
	normalized_img = np.zeros((img.shape[0], img.shape[1]))
	normalized_img = cv2.normalize(img, normalized_img, 0, 255, cv2.NORM_MINMAX)
	return normalized_img

def create_gaussian_pyramid(image, gaussian, depth=5):
	"""
	Creates a gaussian pyramid for a given image
	Parameters:
	image: An n x m x 3 numpy array representing the image
	gaussian: an n x n numpy array representing the Gaussian kernel to apply
	depth: The depth of the pyramid, i.e. how many times to downsample
	Returns:
	An array containing the downsampled images
	"""
	scaled_images = [image]

	for level in range(0, depth):
		sampled_image = subsample_image(scaled_images[level], gaussian)
		scaled_images.append(sampled_image)

	return scaled_images

def subsample_image(image, gaussian, sample_rate=2):
	"""
	Subsamples an image
	Parameters:
	image: An n x m x 3 numpy array representing the image
	gaussian: an n x n numpy array representing the Gaussian kernel to apply
	sample_rate: The rate at which to sample the image by
	Returns:
	An array containing the downsampled image
	"""
	# Apply low pass filter to image
	image_blur = built_in_convolution(image, gaussian)

	# Get image dimensions
	image_height, image_width = image_blur.shape[0], image_blur.shape[1]

	# Create an array to represent the sub-sampled image, i.e. n/2 x m/2 x 3
	subsampled_image = np.zeros((image_height // 2, image_width // 2, 3), dtype=np.uint8)

	i = 0
	# Sample the original image at the given rate
	for x in range(0, image_height, sample_rate):
		j = 0
		for y in range(0, image_width, sample_rate):
			# Sample blurred image at specified sample rate
			subsampled_image[i, j] = image_blur[x, y]
			j += 1
		i += 1

	return subsampled_image

def rotate_image(image, angle, adjust_boundaries=True):
	"""
	Rotates an image
	Parameters:
	image: An n x m x 3 numpy array representing the image
	angle: The angle in degrees by which to rotate the image
	adjust_boundaries: Whether or not to adjust the image boundaries to prevent cut off
	Returns:
	An array representing the rotated image
	"""

	# Get image dimensions
	image_height, image_width = image.shape[0], image.shape[1]
	image_center = (image_width // 2, image_height // 2)

	# Get the rotation matrix for the image
	rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1)

	# Compute new dimension boundaries
	abs_cos = abs(rotation_matrix[0, 0]) 
	abs_sin = abs(rotation_matrix[0, 1])

	# Compute new image boundaries
	horizontal_bound = int((image_height * abs_sin) + (image_width * abs_cos))
	vertical_bound = int((image_height * abs_cos) + (image_width * abs_sin))

	# Realign image centre
	rotation_matrix[0, 2] += horizontal_bound / 2 - image_center[0]
	rotation_matrix[1, 2] += vertical_bound / 2 - image_center[1]

	# Rotate the image with computed matrix
	rotated_image = cv2.warpAffine(image, rotation_matrix, (horizontal_bound, vertical_bound))

	return rotated_image

def get_rotated_square_coordinates(angle, side_length):
	"""
	Gets the coorindates of the rotated square
	Parameters:
	angle: The angle to rotate the square by in degrees
	side_length: The length of the side of the square
	Returns:
	An array of two value tuples representing the square corners from top left to bottom left clockwise
	"""
	angle = int(angle)

	# Ensure that the angle of rotation is between 0 and 90 degrees
	relative_angle = math.radians(0)
	if angle < 90:
		relative_angle = math.radians(angle)
	elif 90 <= angle < 180:
		relative_angle = math.radians(angle - 90)
	elif 180 <= angle < 270:
		relative_angle = math.radians(angle - 180)
	elif 270 <= angle < 360:
		relative_angle = math.radians(angle - 270)

	# Compute rotation factor
	b = (side_length) / (1 + math.tan(relative_angle))
	a = int(side_length - b)

	# Construct new coordinates
	new_coordinates = [
		(a, side_length), 
		(side_length, side_length - a), 
		(side_length - a, 0), 
		(0, a)
	]
	
	return new_coordinates

def non_maxima_suppression(box_configs, threshold):
	"""
	Performs a non-maxima suppression strategy on the bounding boxes
	Parameters:
	box_configs: 
	threshold: The IOU threshold
	Returns:
	An array containing the selected box configurations
	"""

	# Get array of box confidence scores
	scores = np.array([t[3] for t in [b for b in box_configs]])

	# Get array of box coordinates
	boxes = np.array([t[0] for t in [b for b in box_configs]])

	# Every box must have an associated confidence score
	if len(boxes) != len(scores):
		return []

	if len(boxes) == 0 or len(scores) == 0:
		return []
	
	# Get coordinates of boxes
	boxes = boxes.astype("float")
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# Calculate area of each box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	
	# Sort boxes by confidence scores desc
	ranked_boxes = np.argsort(scores)

	# Store list of candidate boxes
	candidate_boxes = []

	# Iterate through each box, removing ones which exceed the IOU of the most confident pick
	while len(ranked_boxes) != 0:
		# Iteratively look at each box
		candidate_index = len(ranked_boxes) - 1
		candidate_box = ranked_boxes[candidate_index]

		# Add this box to the list of candidate boxes
		candidate_boxes.append(candidate_box)

		# Get a list of all other boxes to compare
		compare_boxes = ranked_boxes[:candidate_index]

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		x1_max = np.maximum(x1[candidate_box], x1[compare_boxes])
		y1_max = np.maximum(y1[candidate_box], y1[compare_boxes])
		x2_min = np.minimum(x2[candidate_box], x2[compare_boxes])
		y2_min = np.minimum(y2[candidate_box], y2[compare_boxes])

		# Calculate box dimensions >= 0
		width = np.maximum(0, x2_min - x1_max + 1)
		height = np.maximum(0, y2_min - y1_max + 1)

		# Compute the overlap between the candidate box and every other box
		overlap = (width * height) / area[compare_boxes]

		# Remove boxes which have an overlap exceeding the specified threshold
		eliminate = np.concatenate(([candidate_index], np.where(overlap > threshold)[0]))
		ranked_boxes = np.delete(ranked_boxes, eliminate)

	return [box_configs[b] for b in candidate_boxes]
