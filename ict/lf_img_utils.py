#!/usr/bin/env python

import cStringIO
from numpy import ceil, floor, array, asarray
from PIL import Image
import cv, cv2


#DEFAULT_RESIZE_SMOOTHING_MODE = Image.NEAREST # This is the default
DEFAULT_RESIZE_SMOOTHING_MODE = Image.ANTIALIAS # This is supposed to look the best. (But could be slower.)
def numpyRGBFromEntryString(db_entry_image,max_size_dim=None):
  return cv2ImageFromEntryString(db_entry_image,max_size_dim,True)

def cv2ImageFromEntryString(db_entry_image,max_size_dim=None,retain_rgb_ordering=False):
  """
  Additional conversion utility, allowing us to obtain 'new style' open cv images.
  (These are basically uint8 numpy array-like objects, with channels in BGR order.)

  Input can be the the base64 encoded string, as typically stored in mongo (e.g. entry['cimage'])
  or a simple string (e.g. str(entry['cimage'])) -- we do the string casting internally too,
  just in case.
  """
  # Force string conversion again (so we can use raw entry or string)
  img_str = str(db_entry_image)
  sio_img = cStringIO.StringIO(img_str)
  pil_img = Image.open(sio_img).convert("RGB") # Ensure RGB conversion so that we know what to deal with
  if max_size_dim is not None:
    pil_img = resizeAspectPil(pil_img,max_size_dim)
  cv_img = cv.CreateImageHeader(pil_img.size, cv.IPL_DEPTH_8U, 3)
  cv.SetData(cv_img, pil_img.tostring(), pil_img.size[0]*3)
  if not retain_rgb_ordering:
    cv.CvtColor(cv_img, cv_img, cv.CV_RGB2BGR)
  cv2_img = asarray(cv.GetMat(cv_img))
  return cv2_img


def pil_to_cv(pil_img):
  ## TODO: Think this is buggy
  pil_img = pil_img.convert('RGB')
  cvImage = cv.CreateImageHeader(pil_img.size, cv.IPL_DEPTH_8U, 3)
  cv.SetData(cvImage, pil_img.tostring(), pil_img.size[0]*3)
  cv.CvtColor(cvImage, cvImage, cv.CV_RGB2BGR)
  return cvImage


#######################################################################
################################ PIL ##################################
#######################################################################
def string_to_pil(img_str):
  """ Load a binary image string into a PIL Image """
  sio_img = cStringIO.StringIO(img_str)
  pil_img = Image.open(sio_img)
  return pil_img

def pil_to_string(pil_img, quality=None, optimize=True):
  """ Write pil_img into a binary JPEG string """
  sio_out_str = cStringIO.StringIO()
  if quality is not None:
    pil_img.save(sio_out_str,'JPEG',quality=quality,optimize=optimize)
  else:
    pil_img.save(sio_out_str,'JPEG',optimize=optimize)
  return sio_out_str.getvalue()

def pilImage2NumpyArray(pilImage, TRAILING_FLAT_CHANNELS=False,
                        BIT_DEPTH_CONVERSION=None):
  """Converts a PIL image object into a numpy array.
  """
  if BIT_DEPTH_CONVERSION is None:
    ## Integer division of an integer 1 will have no effect
    BIT_DEPTH_CONVERSION = 1
  if not TRAILING_FLAT_CHANNELS:
    ## This return version gives arrays suitable for direct display (e.g. with
    ## imshow)
    return array(pilImage)/BIT_DEPTH_CONVERSION
  else:
    ## This return version gives arrays that are like the lookflow data format 
    ## (with shape 3 x im_length)
    return array(pilImage).reshape((-1,3)).transpose() / BIT_DEPTH_CONVERSION
    
def resizeSquareCropPil(pil_img, new_dim=256):
  """ Resize pil image to (new_dim,new_dim) by cropping the largest centered
  square image. """
  crop_box = getCropBoxPil(pil_img.size)
  central_pil_img = pil_img
  if pil_img.size != crop_box:
    central_pil_img = pil_img.crop(crop_box) # Only crop if we have to
  if central_pil_img.size != (new_dim,new_dim): # Only resize if we have to.
    central_pil_img = resizePil(central_pil_img,(new_dim,new_dim))
  if central_pil_img.mode != "RGB":
    central_pil_img = central_pil_img.convert("RGB") # Only convert if we have to
  return central_pil_img

def resizeAspectPil(pil_img, new_dim=256):
  """ Resize pil image to the largest aspect-preserving size that fits within
  (new_dim,new_dim) """
  (orig_width,orig_height) = pil_img.size
  if orig_width == orig_height:  ## Aspect ratio of 1.
    new_size = (new_dim, new_dim)
  elif orig_width > orig_height:
    new_size = (new_dim, int(ceil(new_dim * orig_height/(1.0*orig_width))))
  else:
    new_size = (int(ceil(new_dim * orig_width/(1.0*orig_height))), new_dim)
  resize_img = resizePil(pil_img,new_size)
  return resize_img

def resizePil(pil_img,new_size,resample=DEFAULT_RESIZE_SMOOTHING_MODE):
  if new_size == pil_img.size:
    return pil_img.copy() #No-op
  else:
    return pil_img.resize(new_size,resample=resample)

def getCropBoxPil(img_size):
  """
  Extracts a central, square shaped region from an image.
  """
  (width,height) = img_size
  if width == height:
    return (0,0,width,height)
  if width > height: # Image is wider than tall.
    # Want to symmmetrically cut off the excess on either side of central region
    half_overage = int( (width - height)/2.0 )
    crop_box = (half_overage,0,half_overage+height,height)
    return crop_box
  else: #Image is taller than wide
    half_overage = int( (height - width)/2.0 )
    crop_box = (0,half_overage,width,half_overage+width)
    return crop_box
    

#######################################################################
############################## opencv #################################
#######################################################################
def string_to_cv(img_str,resized=None):
  """ Load a binary image string into a opencv Image """
  ## Inefficient but not sure of a better way
  pil_img = string_to_pil(img_str)
  if resized is not None:
    pil_img = resizeAspectPil(pil_img,resized)
  cv_img = pil_to_cv(pil_img)
  return cv_img

## Depends on our deprecated stitch code:
# def cv_to_string(cv_img):
#   """ Write cv_img into a binary JPEG string """
#   import stitch  # For writing to string
#   outputString = stitch.iplImageToString(cvImage)
#   return outputString
    
def resizeSquareCropCV(cv_img, new_dim=256):
  """ Resize pil image to (new_dim,new_dim) by cropping the largest centered
  square image. """
  (orig_width,orig_height) = (cv_img.width,cv_img.height)
  crop_box = getCropBoxCV((orig_width,orig_height))  
  cv.SetImageROI(cv_img, crop_box)
  new_size = (new_dim,new_dim)
  central_pil_img = cv.CreateImage(new_size, cv_img.depth, cv_img.nChannels)
  cv.Resize(cv_img, central_pil_img)
  ## Cleanup: restore cv_img ROI
  cv.SetImageROI(cv_img, (0,0,orig_width,orig_height))
  return central_pil_img

def resizeAspectCV(cv_img, new_dim=256):
  """ Resize cv image to the largest aspect-preserving size that fits within
  (new_dim,new_dim) """
  (orig_width,orig_height) = (cv_img.width,cv_img.height)
  if orig_width == orig_height:  ## Aspect ratio of 1.
    new_size = (new_dim, new_dim)
  elif orig_width > orig_height:
    new_size = (new_dim, int(ceil(new_dim * orig_height/(1.0*orig_width))))
  else:
    new_size = (int(ceil(new_dim * orig_width/(1.0*orig_height))), new_dim)
  resize_img = cv.CreateImage(new_size, cv_img.depth, cv_img.nChannels)
  cv.Resize(cv_img, resize_img)
  return resize_img


def getCropBoxCV(img_size):
  """
  Extracts a central, square shaped region from an image.
  """
  (orig_width,orig_height) = img_size
  if orig_width > orig_height:
    ## Make [orig_height x orig_height]
    startX = orig_width/2 - orig_height/2
    crop_box = (startX, 0, orig_height, orig_height)
    return crop_box
  else:
    ## Make [orig_width x orig_width]
    startY = orig_height/2 - orig_width/2
    crop_box = (0, startY, orig_width, orig_width)
    return crop_box
