# kaggle-TGS-Salt-Identification-Challenge-with-unet
Kaggle TGS Salt Identification Challenge with unet.
Using data augmentation and batch normalization layers in unet.
Reached 0.687 inline, keep promoting.

the file data_overview may help you know the data and make decisions.

Here records the scores using unet.Other networks would be shown in other repositories.

0.628 - base unet without data augmentation.

0.661 - base unet with data augmentation: veitical and horizonal flip, 90, 180, 270 degree rotation.

0.652 - base unet with data augmentation: veitical and horizonal flip, 90, 180, 270 degree rotation, and with value of pixels random shift.

0.687 - unet with batch normalization layers with data augmentation: veitical and horizonal flip, 90, 180, 270 degree rotation.

0.587 - unet with sample balancing.

0.656 - segnet.

0.715 - Change the order of bn and activation layers.

0.541 - add Gaussian noise, helpless.

0.722 - using crf
