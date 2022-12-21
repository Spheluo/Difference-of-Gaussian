# Difference-of-Gaussian
## Implementation of Difference of Gaussian (DoG)

### 1. Use eval.py to generate DoG images and evaluate the keypoints generated from DoG.py. The command is as follows: 

python3 eval.py --image_path './testdata/1.png' --threshold '5' --gt_path './testdata/1_gt.npy'

### 2. Use main.py to generate figures of different thresholds. The command is as follows:

python3 main.py --image_path './testdata/2.png' --threshold '2'

python3 main.py --image_path './testdata/2.png' --threshold '5'

python3 main.py --image_path './testdata/2.png' --threshold '7'
