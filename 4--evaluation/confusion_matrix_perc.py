'''
NOTE 2020-02-11 - This script is "frozen" and archived as part of Ross Winans's masters thesis work. This script was forked
from https://github.com/svpino/tf_object_detection_cm/blob/master/confusion_matrix.py in mid-2020 and 
minor modifications were made to create the "pretty" confusion matricies seen in Ross's thesis paper.

This script is highly tailored to create plots for the thesis paper (i.e. the plot titles and legend name are
hard coded). A pull request will (should?) be created to simplify/incorporate the pretty_print_cm() funtion
to the original repo.

Special thanks and all credit due to this script's primary author, svpino.
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

from object_detection.core import standard_fields
from object_detection.metrics import tf_example_parser
from object_detection.utils import label_map_util

flags = tf.compat.v1.app.flags

flags.DEFINE_string('label_map', None, 'Path to the label map')
flags.DEFINE_string('detections_record', None, 'Path to the detections record file')
flags.DEFINE_string('output_directory', None, 'Path to the directory to output the resulting csv files.')

FLAGS = flags.FLAGS

IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.2

def compute_iou(groundtruth_box, detection_box):
    g_ymin, g_xmin, g_ymax, g_xmax = tuple(groundtruth_box.tolist())
    d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box.tolist())
    
    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)

def process_detections(detections_record, categories):
    record_iterator = tf.python_io.tf_record_iterator(path=detections_record)
    data_parser = tf_example_parser.TfExampleDetectionAndGTParser()

    confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))

    image_index = 0
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        decoded_dict = data_parser.parse(example)
        
        image_index += 1
        
        if decoded_dict:
            groundtruth_boxes = decoded_dict[standard_fields.InputDataFields.groundtruth_boxes]
            groundtruth_classes = decoded_dict[standard_fields.InputDataFields.groundtruth_classes]
            
            detection_scores = decoded_dict[standard_fields.DetectionResultFields.detection_scores]
            detection_classes = decoded_dict[standard_fields.DetectionResultFields.detection_classes][detection_scores >= CONFIDENCE_THRESHOLD]
            detection_boxes = decoded_dict[standard_fields.DetectionResultFields.detection_boxes][detection_scores >= CONFIDENCE_THRESHOLD]
            
            matches = []
            
            if image_index % 100 == 0:
                print("Processed %d images" %(image_index))
            
            for i in range(len(groundtruth_boxes)):
                for j in range(len(detection_boxes)):
                    iou = compute_iou(groundtruth_boxes[i], detection_boxes[j])
                    
                    if iou > IOU_THRESHOLD:
                        matches.append([i, j, iou])
                    
            matches = np.array(matches)
            if matches.shape[0] > 0:
                # Sort list of matches by descending IOU so we can remove duplicate detections
                # while keeping the highest IOU entry.
                matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
                
                # Remove duplicate detections from the list.
                matches = matches[np.unique(matches[:,1], return_index=True)[1]]
                
                # Sort the list again by descending IOU. Removing duplicates doesn't preserve
                # our previous sort.
                matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
                
                # Remove duplicate ground truths from the list.
                matches = matches[np.unique(matches[:,0], return_index=True)[1]]
                
            for i in range(len(groundtruth_boxes)):
                if matches.shape[0] > 0 and matches[matches[:,0] == i].shape[0] == 1:
                    confusion_matrix[groundtruth_classes[i] - 1][detection_classes[int(matches[matches[:,0] == i, 1][0])] - 1] += 1 
                else:
                    confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1
                    
            for i in range(len(detection_boxes)):
                if matches.shape[0] > 0 and matches[matches[:,1] == i].shape[0] == 0:
                    confusion_matrix[confusion_matrix.shape[0] - 1][detection_classes[i] - 1] += 1
        else:
            print("Skipped image %d" % (image_index))
    
    group_percentages = []
    for row in confusion_matrix:
       for val in row:
            percentage = (val / np.sum(row)) * 100
            group_percentages.append(percentage)
    
    conf_matrix_perc = np.asarray(group_percentages).reshape(confusion_matrix.shape[0], confusion_matrix.shape[1])
    print(conf_matrix_perc)
    print("Processed %d images" % (image_index))

    return confusion_matrix, conf_matrix_perc

def display(confusion_matrix, conf_matrix_percent, categories, output_directory):
    print("\nConfusion Matrix:")
    print(confusion_matrix, "\n")

    results = []
    
    print("\nClass counts:")
    for i in range(len(categories)):
        id = categories[i]["id"] - 1
        name = categories[i]["name"]
        
        total_target = np.sum(confusion_matrix[id,:])
        
        print(f"{name}: {total_target}")

        total_predicted = np.sum(confusion_matrix[:,id])
        
        precision = float(confusion_matrix[id, id] / total_predicted)
        recall = float(confusion_matrix[id, id] / total_target)
        
        #print('precision_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, precision))
        #print('recall_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, recall))
        
        results.append({'category' : name, 'precision_@{}IOU'.format(IOU_THRESHOLD) : precision, 'recall_@{}IOU'.format(IOU_THRESHOLD) : recall})
    
    df = pd.DataFrame(results)
    print(df)
    print(f"\nmAP@0.5IOU: {df['precision_@0.5IOU'].mean()}")
    results_output_path = os.path.join(output_directory, "results.csv")
    df.to_csv(results_output_path)

def pretty_print_cm(confusion_array, confusion_array_perc, categories, output_directory):
    categories_list = [f"{c['name']} ({c['id']})" for c in categories]
    categories_list.append('nothing (N/A)') # we append a 'nothing' class based on the implementation above.

    confusion_matrix = pd.DataFrame(data=confusion_array_perc, index=categories_list, columns=categories_list)
    
    #confusion_array = confusion_matrix.to_numpy()
    confusion_mask = confusion_array == 0

    group_counts = [f"{value}" for value in confusion_array.flatten()]
    group_percentages = [f"{value}" for value in confusion_array_perc.flatten()]
    pretty_group_percentages = []

    for perc in group_percentages:
        pretty_group_percentages.append(f"{float(perc):.2f}%")
                     
    labels = [f"{v1} \n {v2}" for v1,v2 in zip(pretty_group_percentages,group_counts)]
    labels = np.asarray(labels).reshape(confusion_array.shape[0], confusion_array.shape[1])

    fig = plt.figure(figsize=(7.5,9), dpi=600)
    ax = sns.heatmap(confusion_matrix, annot=labels, vmin=0, vmax=100, cmap="YlGnBu",fmt='', square=True, mask=confusion_mask, linewidth=.05, linecolor='gray', annot_kws={"fontsize":8}, cbar_kws={"shrink": .67})
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    plt.title('Class Confusion Matrix for SSD with MobileNet v2', fontname="Arial", fontsize = 12) # title with fontsize 20
    plt.xlabel('Predicted Objects', fontname='Arial', fontsize = 11) # x-axis label with fontsize 15
    plt.ylabel('True Objects', fontname='Arial', fontsize = 11) # y-axis label with fontsize 15

    fig = ax.get_figure()
    fig.savefig(os.path.join(output_directory, '5_3_fr_ir_confusion_matrix.png'), dpi=600, bbox_inches='tight')

def main(argv):
    del argv
    required_flags = ['detections_record', 'label_map', 'output_directory']
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

    label_map = label_map_util.load_labelmap(FLAGS.label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)

    confusion_matrix, conf_mat_perc = process_detections(FLAGS.detections_record, categories)

    display(confusion_matrix, conf_mat_perc, categories, FLAGS.output_directory)    
    
    pretty_print_cm(confusion_matrix, conf_mat_perc, categories, FLAGS.output_directory)
if __name__ == '__main__':
    tf.compat.v1.app.run(main)