import os
import sys

image_directory = 'D:/Dropbox/Aubrey Research/RegionTransfer_Latex/figures/'
#tex_file = 'D:/Dropbox/Aubrey Research/RegionTransfer_LatexRegionTransfer.tex'

files = list(os.listdir(image_directory))
for file in files:
    idx = file.find('-eps-converted-to.pdf')
    if idx == -1:
        continue
    new_name = file[:idx] + '.pdf'
    if new_name not in files:
        os.rename(image_directory + file, image_directory + new_name)