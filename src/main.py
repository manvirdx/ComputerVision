import argparse
import config
import sys
import os
import utils
import cv2
import numpy as np
from convolutions import ds_convolution, built_in_convolution
import template_matching as tm
import time

def main():
	# feature mode
	ap = argparse.ArgumentParser()

	# Add the arguments to the parser
	ap.add_argument("-f", "--feature", required=True, 
					help="The feature to perform: 'convolutions' or 'template_matching'")

	ap.add_argument("-m", "--mode", required=True, 
					help="The mode of operation: 'train' or 'test'")
	
	ap.add_argument("-i", "--image", required=False,
				help="The image to perform the convolution on")

	args = vars(ap.parse_args())

	feature = args['feature']
	mode = args['mode']

	if feature == 'optimise':
		optimise_parameters()

	if feature == config.CONVOLUTIONS_ARG:
		image_dir = args['image']
		if image_dir is None:
			raise Exception("Please enter the path to the image to perform the convolution on.")

		if not os.path.exists(image_dir):
			raise Exception("The image at the specified directory was not found.")

		convolute(image_dir)
	
	if feature == config.TEMPLATE_MATCHING_ARG:
		# Perform template matching
		if mode == config.TRAINING_ARG:
			utils.delete_directory(config.TEMPLATE_OUTPUT_DIR)
			return template_matching(config.TRAINING_DIR, config.TEMPLATE_OUTPUT_DIR)

		image = test_template_matching(config.TESTING_DIR, config.TEMPLATE_OUTPUT_DIR)[0]
		cv2.imshow('test-2', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def optimise_parameters():
	pyramid_levels = [3,4,5]
	# pyramid_levels = [x for x in range(3,6)]
	rotations = [[x for x in range(0, 360, rot)] for rot in range(20,35,5)]
	gaussian_parameters = [[5,5,15]]

	# Create results directory
	utils.create_directory(config.RESULTS_DIR)

	for level in pyramid_levels:
		for rots in rotations:
			for g_n, gaussian in enumerate(gaussian_parameters):
				step_size = rots[1] - rots[0]
				row,col,dev = gaussian
				g = utils.gaussian_kernel(row, col, dev)
				utils.delete_directory(config.TEMPLATE_OUTPUT_DIR)
				print('training rotation {} level {} gaussian {}-{}-{}'.format(step_size,level,row,col,dev), rots, level)

				start = time.time()

				template_matching(config.TRAINING_DIR, config.TEMPLATE_OUTPUT_DIR, level, rots, g)
				new_dir = config.RESULTS_DIR+'level{}-rot{}-g-{}-{}-{}/'.format(level,step_size,row,col,dev)
				utils.create_directory(new_dir)
				print('testing', rots, level)
				images = test_template_matching(config.TESTING_DIR, config.TEMPLATE_OUTPUT_DIR)
				end = time.time()
				time_elapsed = end - start
				utils.write_to_file(new_dir+'time.txt', time_elapsed)
				
				for idx, im in enumerate(images):
					cv2.imwrite(new_dir+'{}.png'.format(idx), im)

	return True

def convolute(image_dir):
	kernels = {
		'identity': np.array([[0,0,0],[0,1,0],[0,0,0]]),
		'v_edge': np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
		'h_edge': np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
		'd_edge': np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]]),
		'gaussian_blur': np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16,
		'gaussian_blur_5x5': np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]) / 256,
		'box_blur': utils.generate_box_blur(3),
		'box_blur_5x5': utils.generate_box_blur(5),
		'sharpen': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
	}

	# Read in image
	image = cv2.imread(image_dir)

	# Guard against invalid image directory
	if image is None:
		raise Exception("The image at the specified directory was not found.")

	# Normalise image RGB values
	image = image.astype(float) / 255.0

	# Define the kernel
	kernel = kernels['box_blur_5x5']

	# Apply convolution to image with written function and built in method
	conv = ds_convolution(image, kernel)
	built_in_conv = built_in_convolution(image, kernel)

	# Display each convolution side by side for comparison
	horizontal_concat = utils.concatenate_images([image, built_in_conv, conv])
	cv2.imshow('Convolutions', horizontal_concat)

	# Prevent window from destroying immediately
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def template_matching(data_dir, template_dir, pyramid_depth=4, rotations=None, gaussian=None):
	if rotations is None:
		rotations = [x for x in range(0, 360, 30)]

	if gaussian is None:
		gaussian = utils.gaussian_kernel(5,5,15)

	image_names = utils.get_files(data_dir, extension='.png')
	for image_name in image_names:
		image = cv2.imread(data_dir + image_name + '.png', cv2.IMREAD_UNCHANGED)
		image_filtered = tm.pre_process_image(image)

		# Create a Gaussian pyramid of given depth for each image
		pyramid = tm.create_gaussian_pyramid(image_filtered, gaussian, pyramid_depth)

		# Create directory for class
		image_class = tm.get_class_name(image_name)
		class_dir = template_dir + image_class + '/'
		utils.create_directory(class_dir)
		# Path(class_dir).mkdir(parents=True, exist_ok=True)

		# Rotate each scaled image
		for scale_index, scaled_image in enumerate(pyramid):
			for angle in rotations:
				rotated_image = tm.rotate_image(scaled_image, angle)

				# Save image to png file
				file_name = image_class + "-level" + str(scale_index) + "-rotation" + str(angle) + ".png"
				cv2.imwrite(class_dir + file_name, rotated_image)

	return True

def test_template_matching(testing_dir, template_dir, threshold=0.5):
	testing_images = utils.get_files(testing_dir, remove_extension=False)[:1]
	classes = os.listdir(template_dir)
	images = []
	# Iterate over testing images
	for image_num, test_image_name in enumerate(testing_images):
		test_image = cv2.imread(testing_dir + test_image_name)
		print('Test image', image_num)
		boxes = []
		# Iterate over each template class
		for image_class in classes:
			print('Testing class', image_class)
			template_class_dir = template_dir + image_class+'/'
			templates = utils.get_files(template_class_dir, remove_extension=False)

			cur_best_template = {
				'class_name': " ",
				'template_name': " ",
				'min_val': 0.0,
				'max_val': 0.0,
				'min_loc': 0.0,
				'max_loc': 0.0
			}

			# Iterate over each template for the given class
			for template_name in templates:
				level, rotation = utils.get_template_config(template_name)
				template = cv2.imread(template_class_dir + template_name)

				w, h = template.shape[:2]
				if w > test_image.shape[0] or h > test_image.shape[1]:
					continue

				res = cv2.matchTemplate(test_image,template,cv2.TM_CCORR_NORMED)
				_, max_val, _, max_loc = cv2.minMaxLoc(res)

				# penalties for smaller images
				if level == 5:
					max_val = max_val * 0.3
				if level == 4:
					max_val = max_val * 0.6
				if level == 3:
					max_val = max_val * 0.7
				elif level == 2:
					max_val = max_val * 0.8
				elif level == 1:
					max_val = max_val * 0.8

				# Choose best match per template
				if max_val > cur_best_template['max_val']:
					cur_best_template = {
						'class_name': image_class,
						'template_name': template_name,
						'res': res,
						'max_val': max_val,
						'max_loc': max_loc,
						'w': w,
						'h': h,
						'rotation': rotation
					}

			# Only consider templates which pass the similarity threshold 
			if cur_best_template['max_val'] > threshold:

				top_left = cur_best_template['max_loc']
				rotation = cur_best_template['rotation']
				height = cur_best_template['h']
				width = cur_best_template['w']
				box = [top_left[0], top_left[1], top_left[0] + width, top_left[1] + height]
				boxes.append({
					'box': box,
					'image_class': cur_best_template['class_name'],
					'template_name': cur_best_template['template_name'],
					'rotation': rotation,
					'height': height,
					'conf': cur_best_template['max_val']
				})

		bounding_boxes = [(b['box'], b['template_name'], b['image_class'], b['conf'], b['height']) for b in boxes]

		# Non-maxima suppression strategy
		points = tm.non_maxima_suppression(bounding_boxes, 0.2)
		
		# Draw boxes around the matched templates
		for point_temp in points:

			level, rotation = utils.get_template_config(point_temp[1])
			point = point_temp[0]
			top_left = (point[0], point[1])
			height = point_temp[4]

			p1, p2, p3, p4 = tm.get_rotated_square_coordinates(rotation, height)
			cv2.line(
				img=test_image,
				pt1=(p1[0] + top_left[0], p1[1] + top_left[1]),
				pt2=(p2[0] + top_left[0], p2[1] + top_left[1]),
				color=250,
				thickness=1
			)
			cv2.line(
				img=test_image,
				pt1=(p2[0] + top_left[0], p2[1] + top_left[1]),
				pt2=(p3[0] + top_left[0], p3[1] + top_left[1]),
				color=250,
				thickness=1
			)
			cv2.line(
				img=test_image,
				pt1=(p3[0] + top_left[0], p3[1] + top_left[1]),
				pt2=(p4[0] + top_left[0], p4[1] + top_left[1]),
				color=250,
				thickness=1
			)
			cv2.line(
				img=test_image,
				pt1=(p4[0] + top_left[0], p4[1] + top_left[1]),
				pt2=(p1[0] + top_left[0], p1[1] + top_left[1]),
				color=250,
				thickness=1
			)

			# Write the name of the class above the box
			image_class = point_temp[2]
			cv2.putText(
				img=test_image,
				text=image_class,
				org=top_left,
				fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=0.75,
				color=(255, 255, 255)
			)

		images.append(test_image) 
		
	return images

if __name__ == "__main__":
	main()