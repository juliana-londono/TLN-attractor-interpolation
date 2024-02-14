# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 08:32:30 2023

code to make a gif from images when you want to see how the learning progresses

@author: julil
"""

#TODO: takes only string name

import glob
from PIL import Image

def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*")]
    frame_one = frames[0]
    frame_one.save("fit_TLN2TLN_feb23n6.gif", format="GIF", append_images=frames,
               save_all=True, duration=500, loop=1)
    
if __name__=="__main__":
    print("inside main")
    make_gif(r"C:\Users\julil\Dropbox\Carina-Juliana\Fruit fly gaits\Python code\fit_TLN2TLN_feb23n6")
