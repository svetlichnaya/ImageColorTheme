import argparse
import binascii
import cv2
import numpy as np
import random

from PIL import Image, ImageDraw
from KMeans import KMeans
from MMCQ import MMCQ
import lf_img_utils

class Palette:

    def __init__(self):
        self.w = 500
        self.h = 500
        self.canvas = Image.new('RGB', (self.w, self.h))
        self.paint = ImageDraw.Draw(self.canvas) 
  
    def extract(self, image_path, max_colors=6, method="kmeans", save_output=True, show_img=True):
        print "path: ", image_path
        source = cv2.imread(image_path)
        if method == "kmeans":
            km = KMeans(source, max_colors)
        else:
            km = MMCQ(source, max_colors)
        theme = km.quantize()
        for i, color in enumerate(theme):
            self.addToCanvas(color, i, max_colors)
        if show_img:
            self.canvas.show()
        if save_output:
            img_name = image_path.split("/")[-1]
            swatch_path = "../swatches/" + img_name
            self.canvas.save(swatch_path)    

    def addToCanvas(self, color, index, max_colors):
        r, g, b = color
        print "R: " , r, " G: ", g, " B: ", b 
        height = self.h / max_colors
        top = index*height
        bottom = (index+1)*height
        self.paint.polygon([(0, top),(0, bottom),(self.w, bottom),(self.w, top)], fill=(b, g, r))

    def fromBase64(self, base64_str, max_colors=6, method="kmeans", img_id=None):
        bin_image = binascii.a2b_base64(base64_str)
        pil_image = lf_img_utils.string_to_pil(bin_image)
        if not img_id:
            img_id = random.randint(1, 1000)
        img_path = "../tmp/" + str(img_id) + ".jpg"
        pil_image.save(img_path)
        self.extract(img_path, max_colors, method, save_output=False)

if __name__ == "__main__":
    p = Palette()
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--image_file", help="path to image file", type=str)
    parser.add_argument("-k", "--k_colors", help="number of colors to extract", default=6, type=int)
    parser.add_argument("-m", "--method", help="method to generate palette", default="kmeans", type=str)
    parser.add_argument("--show", dest="show_image", help="show image interactively", action='store_true')
    parser.add_argument("--save", dest="save_image", help="save image file", action='store_true')
    parser.set_defaults(show_image=False, save_image=False)   
    args = parser.parse_args()
    p.extract(args.image_file, args.k_colors, args.method, args.save_image, args.show_image)  
