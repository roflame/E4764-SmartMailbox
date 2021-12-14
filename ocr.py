# import the necessary packages
import cv2
from imutils.perspective import four_point_transform
import pytesseract
import argparse
import imutils
import re
import numpy as np
import io
from PIL import Image, ImageEnhance, ImageFilter
import datetime
from collections import namedtuple
import os

def ocr_code():
	# empty output folder
	dir = '/home/pi/output'
	for f in os.listdir(dir):
	    os.remove(os.path.join(dir, f))

	restart_code = 0
	print(datetime.datetime.now())

	# in terminal: python ocr.py -i input/mail1.jpg
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image" , required=False, help="path to input image to be OCR'd")
	ap.add_argument("-d", "--debug", type=int, default=1, help="whether or not we are visualizing each step of the pipeline")
	ap.add_argument("-c", "--min-conf", type=int, default=0, help="minimum confidence value to filter weak text detection")

		# check to see if *digit only* OCR should be performed, and if so, update our Tesseract OCR options
		# ap.add_argument("-d", "--digits", type=int, default=1, help="whether or not *digits only* OCR will be performed")
		# if args["digits"] > 0:
		# 	options = "outputbase digits"
		# text = pytesseract.image_to_string(rgb, config=options)

	# load the input image from disk
	args = vars(ap.parse_args())
	if args["image"]:
		orig = cv2.imread(args["image"])
	else:
		orig = cv2.imread("/home/pi/2.jpg") # for our project, the 1.jpg image is the latest image captured

	cv2.imwrite("/home/pi/output/0-original.jpg", orig)

	# resize input image and compute the ratio of the *new* width to the *old* width
	image = orig.copy()
	image = imutils.resize(image, width=600)
	ratio = orig.shape[1] / float(image.shape[1])

	# convert the image to grayscale, blur it, and apply edge detection to reveal the outline of the input image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, 30, 150) # or try cv2.Canny(blurred, 75, 200)

	# save outputs for troubleshooting
	# (NOT USED) if args["debug"] == 1:
	cv2.imwrite("/home/pi/output/1-gray.jpg", gray)
	cv2.imwrite("/home/pi/output/2-blurred.jpg", blurred)
	cv2.imwrite("/home/pi/output/3-edged.jpg", edged)

	# ================================================ IMAGE OUTLINE =============================================

	# detect contours in the edge map, sort them by size (in descending order), and grab the largest contours
	# Use a copy of the image e.g. edged.copy() because findContours alters the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

	# initialize a contour that corresponds to the input image outline
	contours = None

	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if this is the first contour we've encountered that has four vertices, then we can assume we've found the input image
		if len(approx) == 4:
			contours = approx
			break

	height, width, channels = image.shape

	# if the input image contour is empty then our script could not find the outline of the item
	if contours is None:
		# raise Exception(("Could not find receipt outline. Try debugging your edge detection and contour steps."))
		print("\nCould not find outline.") # "Try debugging your edge detection and contour steps."

		# If no contours are found, assume the boundary is the contour so that we have some output
		contours = np.array([[[0, 0]],[[0, height]],[[width, height]],[[width, 0]]], dtype=np.int32)

	# Add a padding to improve OCR on text close to edges
	padding = 5
	contours[0][0][0] = contours[0][0][0] - padding # max(0, contours[0][0][0] - padding)
	contours[0][0][1] = contours[0][0][1] - padding # max(0, contours[0][0][1] - padding)
	contours[1][0][0] = contours[1][0][0] - padding # max(0, contours[1][0][0] - padding)
	contours[1][0][1] = contours[1][0][1] + padding # min(height, contours[1][0][1] + padding)
	contours[2][0][0] = contours[2][0][0] + padding # min(width, contours[2][0][0] + padding)
	contours[2][0][1] = contours[2][0][1] + padding # min(height, contours[2][0][1] + padding)
	contours[3][0][0] = contours[3][0][0] + padding # min(width, contours[3][0][0] + padding)
	contours[3][0][1] = contours[3][0][1] - padding # max(0, contours[3][0][1] - padding)

	print("\nSo we continue assuming the full image needs to be OCR'ed.")
	print("\nContours: \n", contours)
	# print("Contour Shape: ", contours.shape)
	# print(type(contours),contours.dtype)

	# draw the contour of the input image on the image
	outline = image.copy()
	cv2.drawContours(outline, [contours], -1, (0, 255, 0), 2) # -1 signifies drawing all contours
	cv2.imwrite("/home/pi/output/4-outline.jpg", outline)

	# apply a four-point perspective transform to the *original* image to obtain a top-down bird's-eye view of the input image
	card = four_point_transform(orig, contours.reshape(4, 2) * ratio)
	cv2.imwrite("/home/pi/output/5-transformed.jpg", card)

	# convert the input image from BGR to RGB channel ordering
	rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
	cv2.imwrite("/home/pi/output/6-rgb.jpg", rgb)

	# Enhance image to get clearer results from image_to_text
	enhancedimage = Image.open("/home/pi/output/6-rgb.jpg")
	# enhancedimage = enhancedimage.convert('L')
	enhancedimage = enhancedimage.convert("RGBA")
	newimdata = []
	datas = enhancedimage.getdata()

	for item in datas:
	    if item[0] < 220 or item[1] < 220 or item[2] < 220:
	        newimdata.append(item)
	    else:
	        newimdata.append((255, 255, 255))
	enhancedimage.putdata(newimdata)

	enhancedimage = enhancedimage.filter(ImageFilter.MedianFilter()) # a little blur
	enhancer = ImageEnhance.Contrast(enhancedimage)
	enhancedimage = enhancer.enhance(2)
	enhancer = ImageEnhance.Sharpness(enhancedimage)
	enhancedimage = enhancer.enhance(2)
	# Convert image to black and white
	enhancedimage = enhancedimage.convert('1')
	enhancedimage.save("/home/pi/output/7-enhanced.jpg")

	# ================================================ BACKUP ====================================================

		# use Tesseract to OCR the image
		# text_full_for_backup = pytesseract.image_to_string(enhancedimage)

		# print("\nRAW OUTPUT")
		# print("=============")
		# print(text_full_for_backup)
		# backup_text = open('/home/pi/output/backup.txt', 'w+')
		# backup_text.writelines([str(datetime.datetime.now()),"\n"])
		# backup_text.writelines(text_full_for_backup)
		# backup_text.close()

		# # (NOT USED) Clean up text: strip out non-ASCII text from text because OpenCV replaces each unknown character with a ?
		# # text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

	# # ================================= SPLITTING SENDER vs RECEIVER LOCATIONS =====================================

	print("\nTrying to OCR original image.")

	# # create a named tuple which we can use to create locations of the input document which we wish to OCR
	OCRLocation = namedtuple("OCRLocation", ["id", "bbox", "filter_keywords"])
	# # "bbox": Bounding box coordinates use the order [x, y, w, h] where x and y are the top-left coordinates, and w and h are the width and height
	# # "filter_keywords": A list of words that we do not wish to consider for OCR

	while restart_code < 8 : # We are looping our code so that we can try
	# restart_code = 0: Use RGB image
	# restart_code = 1: Use enhanced image
	# restart_code = 2: Rotate RGB image 90 CW
	# restart_code = 3: Rotate RGB image 90 CCW
	# restart_code = 4: Rotate RGB image 180
	# restart_code = 5: Rotate enhanced image 90 CW
	# restart_code = 6: Rotate enhanced image 90 CCW
	# restart_code = 7: Rotate enhanced image 180

		if restart_code == 0:
			enhancedimage = cv2.imread("/home/pi/output/6-rgb.jpg")
		elif restart_code == 1:
			print("OCR failed. Trying again with enhanced image (code ", restart_code, ")\n")
			enhancedimage = cv2.imread("/home/pi/output/7-enhanced.jpg")
		elif restart_code == 2:
			print("OCR failed. Trying again with 90 CW rotation (code ", restart_code, ")\n")
			enhancedimage = cv2.imread("/home/pi/output/6-rgb.jpg")
			enhancedimage = cv2.rotate(enhancedimage, cv2.ROTATE_90_CLOCKWISE)
		elif restart_code == 3:
			print("OCR failed. Trying again with 90 CCW rotation (code ", restart_code, ")\n")
			enhancedimage = cv2.imread("/home/pi/output/6-rgb.jpg")
			enhancedimage = cv2.rotate(enhancedimage, cv2.ROTATE_90_COUNTERCLOCKWISE)
		elif restart_code == 4:
			print("OCR failed. Trying again with 180 rotation (code ", restart_code, ")\n")
			enhancedimage = cv2.imread("/home/pi/output/6-rgb.jpg")
			enhancedimage = cv2.rotate(enhancedimage, cv2.ROTATE_180)
		elif restart_code == 5:
			print("OCR failed. Trying again with 90 CW rotation (code ", restart_code, ")\n")
			enhancedimage = cv2.imread("/home/pi/output/7-enhanced.jpg")
			enhancedimage = cv2.rotate(enhancedimage, cv2.ROTATE_90_CLOCKWISE)
		elif restart_code == 6:
			print("OCR failed. Trying again with 90 CCW rotation (code ", restart_code, ")\n")
			enhancedimage = cv2.imread("/home/pi/output/7-enhanced.jpg")
			enhancedimage = cv2.rotate(enhancedimage, cv2.ROTATE_90_COUNTERCLOCKWISE)
		elif restart_code == 7:
			print("OCR failed. Trying again with 180 rotation (code ", restart_code, ")\n")
			enhancedimage = cv2.imread("/home/pi/output/7-enhanced.jpg")
			enhancedimage = cv2.rotate(enhancedimage, cv2.ROTATE_180)

		cv2.imwrite("/home/pi/output/8-locations.jpg", enhancedimage)
		height, width, channels = enhancedimage.shape

		# define the locations of each area of the document we wish to OCR
		# sender_start_point = (0, 0)
		# sender_end_point = (int(width/2), int(height/4))
		# receiver_start_point = (0, int(height/4))
		# receiver_end_point = (int(width), int(height)) # end point - not distance from start (like bbox)
		OCR_LOCATIONS = [
			OCRLocation("sender", (0, 0, int(width/2), int(height/3)), ["sender", "name", "address"]),
			OCRLocation("receiver", (0, int(height/3), int(width), int(height*2/3)), ["receiver", "name", "address"]),
		]

		# initialize a results list to store the document OCR parsing results
		parsingResults = []

		if restart_code == 1:
			# before you start OCR_locations for rotated image, save a backup of original image in case unable to OCR so we can see what original OCR locations to determine why it failed
			cv2.imwrite("/home/pi/output/8a-RGBlocations.jpg", boundinglocations)
		if restart_code == 2:
			# before you start OCR_locations for rotated image, save a backup of original image in case unable to OCR so we can see what original OCR locations to determine why it failed
			cv2.imwrite("/home/pi/output/8b-enhancedlocations.jpg", boundinglocations)
		if restart_code == 3:
			# before you start OCR_locations for rotated image, save a backup of original image in case unable to OCR so we can see what original OCR locations to determine why it failed
			cv2.imwrite("/home/pi/output/8c-rotatedCWlocations.jpg", boundinglocations)
		if restart_code == 4:
			# before you start OCR_locations for rotated image, save a backup of original image in case unable to OCR so we can see what original OCR locations to determine why it failed
			cv2.imwrite("/home/pi/output/8d-rotatedCCWlocations.jpg", boundinglocations)

		# loop over the locations of the document we are going to OCR
		for loc in OCR_LOCATIONS:
			# extract the OCR ROI from the aligned image
			locationsimage = cv2.imread("/home/pi/output/8-locations.jpg")
			(x, y, w, h) = loc.bbox
			
			# draw outline on main image for reference
			boundinglocations = cv2.rectangle(locationsimage, (x, y), (x + w, y + h), (0, 0, 255), 5) # cv2.rectangle(image, start_point, end_point, color, thickness)
			# save image that shows OCR locations
			cv2.imwrite("/home/pi/output/8-locations.jpg", boundinglocations)

			roi = locationsimage[y:y + h, x:x + w]

			# OCR the ROI using Tesseract
			text = pytesseract.image_to_string(roi)

			# break the text into lines and loop over them
			for line in text.split("\n"):
				# if the line is empty, ignore it
				if len(line) == 0:
					continue
				# convert the line to lowercase and then check to see if the line contains any of the filter keywords (these keywords are part of the *form itself* and should be ignored)
				lower = line.lower()
				count = sum([lower.count(x) for x in loc.filter_keywords])
				# if the count is zero then we know we are *not* examining a text field that is part of the document itself (ex., info, on the field, an example, help text, etc.)
				if count == 0:
					# update our parsing results dictionary with the OCR'd text if the line is *not* empty
					parsingResults.append((loc, line))
		# print(parsingResults)

		# initialize a dictionary to store our final OCR results
		results = {}
		# loop over the results of parsing the document
		for (loc, line) in parsingResults:
			# grab any existing OCR result for the current ID of the document
			r = results.get(loc.id, None)
			# if the result is None, initialize it using the text and location namedtuple (converting it to a dictionary as namedtuples are not hashable)
			if r is None:
				results[loc.id] = (line, loc._asdict())
			# otherwise, there exists an OCR result for the current area of the document, so we should append our existing line
			else:
				# unpack the existing OCR result and append the line to the existing text
				(existingText, loc) = r
				text = "{}\n{}".format(existingText, line)
				# update our results dictionary
				results[loc["id"]] = (text, loc)
		# print(results)

		OCR_text = []
		for (locID, result) in results.items():
			# unpack the result tuple
			(text, loc) = result
			
			# # display the OCR result to our terminal
			# print(loc["id"])
			# print("=" * len(loc["id"]))
			# print("{}\n\n".format(text))

			OCR_text.append(loc["id"])
			OCR_text.append("{}\n\n".format(text))
		# print("\n", OCR_text)

		sender_content = OCR_text[1]
		# print(sender_content)

		receiver_content = OCR_text[3]
		# print(receiver_content)

		# ========================================= REGULAR EXPRESSIONS =============================================

		# test regex with https://regexr.com/
		# regex commands are https://www.w3schools.com/python/python_regex.asp

		# # use regular expressions to parse out names
		# nameExp = r"^[\w'\-,.][^0-9_!¡?÷?¿/\\+=@#$%ˆ&*(){}|~<>;:[\]]{2,}"
		# nameExp = r"\b([A-Z]\w+)\b" # Gets all words
		nameExp = r"\b([A-Z]\w+(?=[\s\-][A-Z])(?:[\s\-][A-Z]\w+)+)\b"
		sender_names = re.findall(nameExp, sender_content)
		# print("\nSender Names: ", sender_names)
		receiver_names = re.findall(nameExp, receiver_content)
		# print("\nReceiver Names: ", receiver_names)

		# # use regular expressions to parse out mailing addresses
		# # mailExp = r"\d{1,4}( \w+){1,5}, (.*), ( \w+){1,5}, (.*), [0-9]{5}(-[0-9]{4})?"
		mailExp = r"\d{1,4} [\w\s'\-,.]{1,} [0-9]{5}?"
		sender_addr = re.findall(mailExp, sender_content)
		# print("\nSender Addr: ", sender_addr)
		receiver_addr = re.findall(mailExp, receiver_content)
		# print("\nReceiver Addr: ", receiver_addr)

		print("\nSender Name: ")
		# # loop over the detected name and print them to our terminal
		# for name in names:
		#	print(name.strip())
		if sender_names:
			sender_name_var = sender_names[0].strip()
		else:
			sender_name_var = "NONE"
		print(sender_name_var)

		print("\nSender Address: ")
		# # loop over the detected mailing addresses and print them to our terminal
		# for addr in mail_addresses:
		#	print(addr.strip())
		if sender_addr:
			sender_addr_var = sender_addr[0]
			restart_code = 8 # so that loop breaks
		else:
			sender_addr_var = "NONE"
			restart_code += 1
		print(sender_addr_var)

		print("\nReceiver Name: ")
		if receiver_names:
			receiver_name_var = receiver_names[0].strip()
		else:
			receiver_name_var = "NONE"
		print(receiver_name_var)

		print("\nReceiver Address: ")
		if receiver_addr:
			receiver_addr_var = receiver_addr[0]
		else:
			receiver_addr_var = "NONE"
		print(receiver_addr_var)
		print("\n")

			# # use regular expressions to parse out phone numbers
			# phoneNums = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
			# # loop over the detected phone numbers and print them to our terminal
			# print("PHONE NUMBERS")
			# for num in phoneNums:
			# 	print(num.strip())

			# # use regular expressions to parse out email addresses
			# emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text)
			# # loop over the detected email addresses and print them to our terminal
			# print("EMAILS")
			# for email in emails:
			# 	print(email.strip())

	# Write outputs to files
	OCR_text_file = open('/home/pi/output/OCR_text.txt', 'w+')
	OCR_text_file.writelines([str(datetime.datetime.now())])
	OCR_text_file.writelines(["\nSender Name: ", sender_name_var])
	OCR_text_file.writelines(["\nSender Address: ", sender_addr_var])
	OCR_text_file.writelines(["\nReceiver Name: ", receiver_name_var])
	OCR_text_file.writelines(["\nReceiver Address: ", receiver_addr_var])
	OCR_text_file.close()

	receiver_text = open('/home/pi/output/receiver.txt', 'w+')
	receiver_text.writelines(receiver_name_var)
	receiver_text.close()

if __name__ == '__main__':
	ocr_code()