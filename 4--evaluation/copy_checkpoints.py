'''
NOTE 2020-02-11 - This is a "frozen" file that is archived to document Ross's Masters Thesis work.
This file will not be updated in the future. 

Users should use the original file located at the link below for future projects as
this file is not maintained.

NOTE 2020-05-06 - This file was originally forked from 
https://github.com/microsoft/CameraTraps/blob/master/detection/detector_training/copy_checkpoints.py on 
6/17/2020 and adapted to work with Ross's marine debris data in TFODAPI v1.15.

copy_checkpoints.py

Run this script with specified source_dir and target_dir while the model is training to make a copy
of every checkpoint (checkpoints are kept once an hour by default and is difficult to adjust)
'''
import time
import os
import shutil
import optparse

#handle our input arguments
parser = optparse.OptionParser()

parser.add_option('-i', '--input_dir',
    action="store", dest="usr_in_dir",
    type='string', help="The tensorflow directory where checkpoints are being actively written.")

parser.add_option('-o', '--out_dir',
    action='store', dest='usr_out_dir',
    type='string', help='The output directory where copied checkpoints are stored.')

parser.add_option('-n', '--n_minutes',
    action='store', dest='usr_n_minutes',
    type='int', default=10,
    help='The number of minutes to wait before looking for a new checkpoint. Default is 10.')

options, args = parser.parse_args()

check_every_n_minutes = options.usr_n_minutes
source_dir = options.usr_in_dir
target_dir = options.usr_out_dir

os.makedirs(target_dir, exist_ok=True)

num_checks = 0
while True:
    num_checks += 1
    print('Checking round {}.'.format(num_checks))

    for f in os.listdir(source_dir):
        # do not copy event or evaluation results
        try:
            if f.startswith('model') or f.startswith('graph'):
                target_path = os.path.join(target_dir, f)
                if not os.path.exists(target_path):
                    _ = shutil.copy(os.path.join(source_dir, f), target_path)
                    print('Copied {}.'.format(f))
        except IOError:
            print('WARNING: Could not copy {}.'.format(f))

    print('End of round {}.'.format(num_checks))

    time.sleep(check_every_n_minutes * 60)