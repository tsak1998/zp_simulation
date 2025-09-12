The existing codebase is a training and inference pipeline for segmentation tasks

The main task that is in some times hardcoded (like the Datasets) is the detection of pronuclei in early embryos.


The task is to refactor the code and make any additions needed to make the project more generanlized.

Our end goal is to be able to train a new model on blastocyst images that can detect:
zona pellucida, inner cell mass and Trophoblast.

the data will be in the data directory in the BLASTOCYST folder with the following structure:
- Images
- GT_ZP
- GT_ICM
- GT_TE