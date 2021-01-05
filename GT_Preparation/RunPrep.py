import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from GT_Preparation.DataPrep import PrepareGT

# ==============================================================================================
# Script to create videos from the GT-frame collection. Please specify input and output paths.
# Set annotate to True in order to display bounding boxes in the video
# ==============================================================================================

# Define input and output path
input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'dataset', 'frames'))
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'dataset', 'videos', 'generated'))
video_name = "gt_annotated.mp4"
annotate = True

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_prep = PrepareGT(input_path, output_path, video_name)
data_prep.createvid(annotate)