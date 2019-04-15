# Copyright 2018 by Frank Trampe.
# Preprocess to monochrome and get vectors from img2vec.
# Inputs for training include the positive directory, the negative directory, and the training data destination.

import numpy as np
import math
import sys
import argparse
import os
import random
import pickle
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.svm
from img_to_vec import Img2Vec
from PIL import Image

# Initialize img2vec. This triggers downloading the neural net model if not present.
img2vec = Img2Vec(cuda=False, model='resnet-18')

def image_make_rgb(img):
	# https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
	# img must be loaded first.
	flatimg = Image.new("RGB", img.size, (255, 255, 255))
	splitimg = img.split()
	if len(splitimg) > 3:
		flatimg.paste(img, mask=img.split()[3])
	else:
		flatimg.paste(img)
	return flatimg

def get_vec_from_file(fpath):
	# Read in an image from the path and get a vector from it.
	img = Image.open(fpath)
	img.load()
	return img2vec.get_vec(image_make_rgb(img))

def train_or_test_on_images(model, mode, negative_directory, positive_directory):
	# List the directories and flag the y values.
	flist_neg = map(lambda f : [negative_directory + f, 0], [f for f in os.listdir(negative_directory)])
	flist_pos = map(lambda f : [positive_directory + f, 1], [f for f in os.listdir(positive_directory)])
	# Vectorize and shuffle.
	training_data_vectorized = list(map(lambda fv : [get_vec_from_file(fv[0]), fv[1]], list(flist_neg) + list(flist_pos)))
	random.shuffle(training_data_vectorized)
	# Split x and y into separate arrays.
	training_data_separated = [list(map(lambda fv : fv[0], training_data_vectorized)), list(map(lambda fv : fv[1], training_data_vectorized))]
	# Normalize.
	training_data_normalized = [sklearn.preprocessing.normalize(training_data_separated[0]), training_data_separated[1]]
	# Train if specified.
	skmodel = model
	if (not skmodel):
		# skmodel = sklearn.linear_model.LogisticRegression()
		skmodel = sklearn.svm.SVC(probability = True)
		skmodel.fit(training_data_normalized[0], training_data_normalized[1])
	# Score.
	skscore = skmodel.score(training_data_normalized[0], training_data_normalized[1])
	# Report.
	print("Accuracy: %.3f." % skscore)
	if mode == 0:
		return skmodel
	elif mode == 1:
		return skscore
	return None

def predict_on_image_file_with_logic(model, fpath, slice_width, slice_height, opts):
	rv = None
	img = Image.open(fpath)
	img.load()
	imgw = img.size[0]
	imgh = img.size[1]
	slices = []
	if slice_width is None and slice_height is None:
		# No slices, so we just work on the whole image.
		slices += [img]
	else:
		# Set slice sizes if unset.
		slice_width_t = slice_width
		if slice_width_t is None:
			slice_width_t = imgw
		if type(slice_width_t) == str:
			slice_width_t = int(slice_width_t)
		slice_height_t = slice_height
		if slice_height_t is None:
			slice_height_t = imgh
		if type(slice_height_t) == str:
			slice_height_t = int(slice_height_t)
		slices = []
		# Cut the image into slices.
		slice_pos_y = 0
		while slice_pos_y * slice_height_t < imgh:
			slice_pos_x = 0
			while slice_pos_x * slice_width_t < imgw:
				cropbox = [slice_pos_x * slice_width_t, slice_pos_y * slice_height_t, \
				min((slice_pos_x + 1) * slice_width_t, imgw), min((slice_pos_y + 1) * slice_height_t, imgh)]
				slices += [img.crop(cropbox)]
				slice_pos_x += 1
			slice_pos_y += 1
	# Run predictions.
	predictor = model.predict_proba
	if opts["classify"]:
		predictor = model.predict
	slice_results = predictor(list(map(lambda tslice: img2vec.get_vec(image_make_rgb(tslice)), slices)))
	# print(slice_results)
	if slice_results is None or len(slice_results) < len(slices):
		return None
	# Apply thresholds.
	if opts["slice_threshold"] is not None and not opts["classify"]:
		slice_results = list(map(lambda tv: tv[1] > float(opts["slice_threshold"]), slice_results))
	# print(slice_results)
	# Invert individual results.
	if opts["slice_invert"] and (type(opts["slice_invert"]) == int or int(opts["slice_invert"])):
		slice_results = list(map(lambda tv: not tv, slice_results))
	# Run an or filter on all results to get a composite result.
	rv = False
	positive_count = len(list(filter(lambda tv: tv, slice_results)))
	if positive_count:
		rv = True
	# Check for an absolute threshold. If present, check that the count meets it.
	if type(opts["slice_count_threshold_proportional"]) is not None and \
	float(opts["slice_count_threshold_proportional"]) * len(slices) > positive_count:
		rv = False
	# Check for a proportional threshold. If present, check that the count meets it.
	if type(opts["slice_count_threshold"]) is not None and \
	float(opts["slice_count_threshold"]) > positive_count:
		rv = False
	# Invert if requested.
	if opts["group_invert"] and (type(opts["group_invert"]) == int or int(opts["group_invert"])):
		rv = not rv
	return int(rv)

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--train-negative", required=False, help="directory of training images with desired negative output")
	ap.add_argument("--train-positive", required=False, help="directory of training images with desired positive output")
	ap.add_argument("--test-negative", required=False, help="directory of test images with desired negative output")
	ap.add_argument("--test-positive", required=False, help="directory of test images with desired positive output")
	ap.add_argument("--predict-file", required=False, help="an image on which to predict using the model")
	ap.add_argument("--classify", required=False, help="whether to classify rather than give a probability")
	ap.add_argument("--slice-width", required=False, help="width of slices on which to perform the prediction")
	ap.add_argument("--slice-height", required=False, help="height of slices on which to perform the prediction")
	ap.add_argument("--slice-threshold", required=False, default=0.5, help="probability cut-off per slice")
	ap.add_argument("--slice-count-threshold", required=False, default=0.5, help="positive slice count threshold")
	ap.add_argument("--slice-count-threshold-proportional", required=False, default=0.5, help="positive slice count threshold relative to total")
	ap.add_argument("--slice-result-invert", required=False, help="whether to invert the result of each slice before composing")
	ap.add_argument("--final-result-invert", required=False, help="whether to invert the final result")
	ap.add_argument("--model-input", required=False, help="a model to load")
	ap.add_argument("--model-output", required=False, help="a model to save")
	args = vars(ap.parse_args())
	model = None
	score = None
	if args["train_negative"] is not None and args["train_positive"] is not None:
		model = train_or_test_on_images(None, 0, args["train_negative"], args["train_positive"])
	elif args["model_input"] is not None:
		modelfile = open(args["model_input"], "rb")
		model = pickle.load(modelfile)
		modelfile.close()
	if args["test_negative"] is not None and args["test_positive"] is not None:
		score = train_or_test_on_images(model, 1, args["test_negative"], args["test_positive"])
	if args["predict_file"] is not None:
		predict_opts = {"slice_invert": args["slice_result_invert"], "group_invert": args["final_result_invert"], \
		"classify": args["classify"], "slice_threshold": args["slice_threshold"], \
		"slice_count_threshold": args["slice_count_threshold"], \
		"slice_count_threshold_proportional": args["slice_count_threshold_proportional"]}
		print(predict_on_image_file_with_logic(model, args["predict_file"], args["slice_width"], args["slice_height"], predict_opts))
	if args["model_output"] is not None:
		modelfile = open(args["model_output"], "wb")
		pickle.dump(model, modelfile)
		modelfile.close()

main()

