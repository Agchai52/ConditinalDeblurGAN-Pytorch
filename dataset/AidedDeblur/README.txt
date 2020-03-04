## Overview
This is a synthetic dataset for inertial sensors aided deep image deblurring neural network. It contains two subfolders: train/ and test/. There are 2264 sets of data in the train dataset and 1221 in the test. 

## Data in each set:
 In the train or test folder, like "Dataset/train/"
 - "ImageName_ref.png": the groundtruth sharp image
 - "ImageName_blur_ori.png": the blurry image **without** error effects
 - "ImageName_blur_err.png": the blurry image **with error** effects
 - "ImageName_IMU_ori.txt": the gyro and acc data **without error** effects (31 samples)
 - "ImageName_IMU_err.txt": the gyro and acc data **with error** effects (220 samples)
 - "ImageName_param.txt": the parameters used in original data and error data

## For more information, please check the paper:

S. Zhang, A Zhen and R. L. Stevenson, "A Dataset for Deep Image Deblurring Aided by Inertial Sensor Data", EI2020.
