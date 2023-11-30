import numpy as np
import cv2
videofile=r'd:\ND250fpsOct23\Cas9-B1\P1000360.MP4'
outfile=r'd:\ND250fpsOct23\Cas9-B1\P1000360_stab.mp4'

from vidstab import VidStab
import matplotlib.pyplot as plt

stabilizer = VidStab()
stabilizer.stabilize(input_path=videofile, output_path=outfile)

stabilizer.plot_trajectory()
plt.show()

stabilizer.plot_transforms()
plt.show()