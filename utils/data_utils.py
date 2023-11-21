import os
import subprocess
import random
import matplotlib.pyplot as plt

from PIL import Image

def download_lung_semantic_data(path='data'):
	"""Download lung semantic segmentation dataset from Kaggle.
	
	Parameters
	----------
	path : str
		Path to download data to."""
	
	try:
		subprocess.call('scripts/download_data.sh ' + path, shell=True) # running script to download data in shell
	except subprocess.CalledProcessError as e:
		print('Error downloading data: ' + e.output)

def display_image_and_mask(num_images=3):
	"""Display images and masks from the dataset in a plot.

	Parameters
	----------
	num_images : int
		Number of images to display."""
	img = []
	mask = []
	indices = random.sample(range(1000), num_images)
	for index in indices:
		img.append(Image.open(f'./data/train/image/cxrimage_{index}.png').convert('L'))
		mask.append(Image.open(f'./data/train/mask/cxrmask_{index}.jpeg').convert('L'))

	fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

	# Iterate through images and masks
	for i, (img, mask) in enumerate(zip(img, mask)):
		# Plot image
		axes[i, 0].imshow(img, cmap='gray')
		axes[i, 0].set_title(f'Image {i + 1}', fontsize=14, color='blue')  # Set title with increased font size and color

		# Plot mask
		axes[i, 1].imshow(mask, cmap='gray')
		axes[i, 1].set_title(f'Mask {i + 1}', fontsize=14, color='green')  # Set title with increased font size and color

	# Adjust layout
	plt.tight_layout()

	# Display the plot
	plt.show()
	
	

if __name__ == '__main__':
	print('passed')