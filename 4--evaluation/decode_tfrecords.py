'''
NOTE 2020-02-11 - This is a "frozen" file that is archived to document Ross's Masters Thesis work.
This file will not be updated in the future. Any future work on this script will occur at the 
mapping_marine_debris project GitHub (https://github.com/rosswin/mapping_marine_debris).

decode_tfrecords.py
---
This script takes a TFODAPI detections.record file (or multiple detections.record files) and plots the model's detections
along side the ground truth labels (which are conveniently stored alongside the detection by default). If using the 
"multi-plot" mode then the second set of detection will be plotted along the first.

This script also does some nice unioning of the detections and labels to produce publication-quality plots.

Things to know:
1) Only tested with up to n=2 detections.record files. However, this should scale.
2) The ground truth is pulled from the first detections.record.
3) This script pulls the "intersection" of the detections.record files and matches them by image name.
4) This script does some fancy work to make sure that one legend represents all detections.record.
5) Currently pegged to produce high-quality 600dpi images at 3.25 x 3.25 inches
'''


import tensorflow.compat.v1 as tf
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as font_manager
import numpy as np
import os
import sys

ir_tfrecord = r"/mnt/ssd/MASTERS-THESIS-LOCAL/Results/20201025_final_results/incept_resnet/results/detections.record"
mn_tfrecord = r"/mnt/ssd/MASTERS-THESIS-LOCAL/Results/20201025_final_results/mobilenet/results/detections.record"
image_dir = r"/mnt/ssd/2015_dar_coastal/01_dl_images"
output_dir = r"/mnt/ssd/MASTERS-THESIS-LOCAL/Results/20201025_final_results/collages_2021_2"
in_model = r"Manual Interpretation"

# Final Plot's Color and Label Settings
def color_map(in_val):
    if in_val == 1:
        return 'b'
    elif in_val == 2:
        return 'r'
    elif in_val == 3:
        return 'g'
    elif in_val == 4:
        return 'm'
    elif in_val == 5:
        return 'c'
    elif in_val == 6:
        return 'y'
    elif in_val == 7:
        return 'darkorange'
    elif in_val == 8:
        return 'limegreen'
    elif in_val == 9:
        return 'yellow'
    else:
        return 'lightgray'

def label_map(in_val):

    if in_val == 1:
        return 'buoy'
    elif in_val == 2:
        return 'orange foam'
    elif in_val == 3:
        return 'plastic'
    elif in_val == 4:
        return 'net/cloth/bundled line'
    elif in_val == 5:
        return 'line fragment'
    elif in_val == 6:
        return 'metal'
    elif in_val == 7:
        return 'tire'
    elif in_val == 8:
        return 'processed wood'
    elif in_val == 9:
        return 'vessel'
    else:
        return None

def convert_bbox_coord(coord, im_height_or_width=512):
    """ Simple function designed to be used in a pd.apply() statement. Converts TF's 
      preferred format for bounding box coordinates (float percent to PASCAL VOC's 
      preferred format (pixel coordinates, top left origin). 
      
      NOTE: this currently assumes images are of equal height or width OR the user
        has explicitly provided the proper im_height_or_width."""
    return int(coord * im_height_or_width)

def read_and_decode_detections_record_to_pd(input_tfrecord, ground_truth=False, score_threshold=0.2):
    """ Pulls the bounding boxes from a detections.tfrecord and returns a nicely 
        formatted Pandas dataframe with columns containing:associated image filename, 
        class label, xmin, ymin, xmax, ymax. 
        
        Specifying ground_truth=True will pull the ground truth bounding boxes."""

    example= tf.train.Example()
    example_dfs = []
    for record in tf.python_io.tf_record_iterator(input_tfrecord):
        example.ParseFromString(record)
        f = example.features.feature
        
        if ground_truth==True:
            print("Extracting Ground Truth BBoxes...")

            labels = f['image/object/class/label'].int64_list.value
            # need to create a detection_scores column to match the detection bboxes, which would contain actual scores
            detection_scores = [100] * len(labels)

            xmins = f['image/object/bbox/xmin'].float_list.value
            ymins = f['image/object/bbox/ymin'].float_list.value
            xmaxs = f['image/object/bbox/xmax'].float_list.value
            ymaxs = f['image/object/bbox/ymax'].float_list.value
        elif ground_truth==False:
            print("Extracting Detection BBoxes...")

            detection_scores = f['image/detection/score'].float_list.value
            labels = f['image/detection/label'].int64_list.value

            xmins = f['image/detection/bbox/xmin'].float_list.value
            ymins = f['image/detection/bbox/ymin'].float_list.value
            xmaxs = f['image/detection/bbox/xmax'].float_list.value
            ymaxs = f['image/detection/bbox/ymax'].float_list.value
        else:
            print("Invalid ground_truth value. This should be a True or False value.")
            sys.exit(1)

        # float img coordinates to pixel img coordinates
        # NOTE: this code assumes img_dims are equal.
        px_xmins = [convert_bbox_coord(x) for x in xmins] 
        px_ymins = [convert_bbox_coord(y) for y in ymins]
        px_xmaxs = [convert_bbox_coord(x) for x in xmaxs]
        px_ymaxs = [convert_bbox_coord(y) for y in ymaxs]

        # This 5-liner is the best I've whipped up so far to convert from TF's byte_list format
        # to a standard Python string. Gotta be a better way...
        num_detections = len(labels)
        filename = tf.compat.as_str_any(f['image/filename'].bytes_list.value)
        filename2 = filename.lstrip("['b")
        filename3 = filename2.rstrip("']")
        filenames = [filename3] * num_detections

        # zip up all our final columns into a dictionary
        example_dict = {'filename': filenames, 
                        'detection_score': detection_scores, 
                        'detection_label': labels,
                        'xmin': px_xmins,
                        'ymin': px_ymins,
                        'xmax': px_xmaxs,
                        'ymax': px_ymaxs}
        
        # send the dictionary to a Pandas DF, append that DF to a list of DFs to smush into one big DF at the end
        example_df = pd.DataFrame.from_dict(example_dict)
        thinned_example_df = example_df[example_df['detection_score'] >= score_threshold]
        example_dfs.append(thinned_example_df)

    # smush the list of DFs into the final DF containing all detections (or ground truth)
    concat_df = pd.concat(example_dfs, ignore_index=True)
    return concat_df

def plot_multiple_image_with_bboxes(input_dataframes):
    # extract each data frames' "filename" column into a pd.Series, convert those to sets, and take the intersection
    # of those sets. Scales infinitly with the number of pd.DataFrames in input_dataframes
    unique_files = set.intersection(*[set(f['filename']) for f in input_dataframes])
    print(f"{len(unique_files)} unique images to plot")

    for uf in unique_files:
        input_img = os.path.join(image_dir, uf)
        im = np.array(Image.open(input_img), dtype=np.uint8)

        fig, ax = plt.subplots(1, len(input_dataframes), sharex=True, sharey=True, figsize=(8.5, 2.5))
        
        # Left - Ground Truth
        ax[0].imshow(im)
        ax[0].tick_params(
            axis='both',
            which='both',
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False)
        ax[0].set_title("Manual Interpretation")
        gt_bboxes = input_dataframes[0][input_dataframes[0]['filename'] == uf]
        for idx, row in gt_bboxes.iterrows():
            width = int(row['xmax'] - row['xmin'])
            height = int(row['ymax'] - row['ymin'])
            #print(width, height)
            ax[0].add_patch(patches.Rectangle((row['xmin'], row['ymin']), width, height, 
                                            fill=False, color=color_map(row['detection_label']), 
                                            label=label_map(row['detection_label'])))
        ax[1].imshow(im)
        ax[1].tick_params(
            axis='both',
            which='both',
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False)
        ax[1].set_title("Faster R-CNN \n with Inception ResNet v2")
        ir_bboxes = input_dataframes[1][input_dataframes[1]['filename'] == uf]
        for idx, row in ir_bboxes.iterrows():
            width = int(row['xmax'] - row['xmin'])
            height = int(row['ymax'] - row['ymin'])
            #print(width, height)
            ax[1].add_patch(patches.Rectangle((row['xmin'], row['ymin']), width, height, 
                                            fill=False, color=color_map(row['detection_label']), 
                                            label=label_map(row['detection_label'])))
        ax[2].imshow(im)
        ax[2].tick_params(
            axis='both',
            which='both',
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False)
        ax[2].set_title("SSD with MobileNet v2")
        mn_bboxes = input_dataframes[2][input_dataframes[2]['filename'] == uf]
        for idx, row in mn_bboxes.iterrows():
            width = int(row['xmax'] - row['xmin'])
            height = int(row['ymax'] - row['ymin'])
            #print(width, height)
            ax[2].add_patch(patches.Rectangle((row['xmin'], row['ymin']), width, height, 
                                            fill=False, color=color_map(row['detection_label']), 
                                            label=label_map(row['detection_label'])))

        # Single Legend for Multiple Subplots
        leg_font = font_manager.FontProperties(family='Arial')
        handles_list=[]
        labels_list=[]
        for ax in fig.axes:
            handles, labels = ax.get_legend_handles_labels()
            handles_list.extend(handles)
            labels_list.extend(labels)
        by_label = dict(zip(labels_list, handles_list))
        fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.1), ncol=min([5, len(by_label)]), loc='upper center', prop=leg_font, fancybox=False, shadow=False)
        
        #plt.show()
        out_fig = os.path.join(output_dir, f"{uf}.png")
        plt.savefig(out_fig, dpi=600, bbox_inches='tight')

def plot_single_image_with_bboxes(input_dataframe, image_dir, plot_legend=False):
    # Get a list of unique images from the data frame
    unique_files = input_dataframe.filename.unique()
    print(f"{len(unique_files)} images to plot.")
    
    for uf in unique_files:
        input_img = os.path.join(image_dir, uf)
        im = np.array(Image.open(input_img), dtype=np.uint8)

        fig,ax = plt.subplots(1, figsize=(3.25, 3.25))
        
        plt.imshow(im)

        current_bboxes = input_dataframe[input_dataframe['filename'] == uf]
        #print(current_bboxes.head())
        for idx, row in current_bboxes.iterrows():
            width = int(row['xmax'] - row['xmin'])
            height = int(row['ymax'] - row['ymin'])
            #print(width, height)
            ax.add_patch(patches.Rectangle((row['xmin'], row['ymin']), width, height, 
                                            fill=False, color=color_map(row['detection_label']), 
                                            label=label_map(row['detection_label'])))
        
        if plot_legend==True:
            leg_font = font_manager.FontProperties(family='Arial')

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0,-0.2), ncol=len(by_label), loc='lower left', prop=leg_font)

        plt.title(f"{in_model}", fontname="Arial", fontsize=12)
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False
        )
        
        #plt.show()
        out_fig = os.path.join(output_dir, f"{uf}.png")
        plt.savefig(out_fig, dpi=600, bbox_inches='tight')

#single plot
#my_df = read_and_decode_detections_record_to_pd(input_tfrecord, True)
#plot_single_image_with_bboxes(my_df, image_dir, True)  

#multi plot
all_dfs = []
gt_df = read_and_decode_detections_record_to_pd(ir_tfrecord, True)
all_dfs.append(gt_df)
ir_df = read_and_decode_detections_record_to_pd(ir_tfrecord, False)
all_dfs.append(ir_df)
mn_df = read_and_decode_detections_record_to_pd(mn_tfrecord, False)
all_dfs.append(mn_df)

plot_multiple_image_with_bboxes(all_dfs)

