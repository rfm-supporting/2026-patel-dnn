# 2026-patel-dnn
Supporting information for:

>[Patel, J., Flachot, F., Vazquez-Corral, J,. Brainard, D. H., Wallis, T. S. A., Brubaker, M. A., & Murray, R. F. (2026).](https://doi.org/10.1167/jov.26.3.14) A deep convolutional neural network trained for lightness constancy is susceptible to lightness illusions. *Journal of Vision, 26*(3):14, 1-17.

Here we describe the contents of each subdirectory. See files called instructions.txt in the subdirectories for details of how to run the individual experiments and analyses. The code is a mix of Python and MATLAB.

In the top-level directory is venv.yaml, which specifies the Python modules needed to run the code. The Anaconda distribution of Python can use this file to create an appropriate environment.

1. Experiment with human observers

- humans
-- Python code to run the experiment for human observers
-- Python analysis code that quantifies the illusion for each observer

humans/data
- trial-by-trial data files for each observer
- this is the human data reported in the paper

humans/illusions_isoluminant
- illusion images used as stimuli in the experiment
- luminance and reflectance images are provided for each illusion
- in the luminance images, reference and match regions are isoluminant
- files with 'mask' in their name provide masks that make it possible to select the reference and match regions of the corresponding illusion image

humans/tools
- calibration data and custom Python modules used by the experiment script


2. IRNet network

irnet.py
- definition of IRNet network

irnet/render
- Python code to generate training images
- The training data takes up many gigabytes, so we have not posted it here. If you would like help using the code to generate the training data, feel free to contact us.

irnet/train
- Python code to carry out training of IRNet

irnet/model_weights
- weights resulting from ten training runs

irnet/apply
- Python code to apply IRNet with trained weights to stimulus images

irnet/data
- Python code to select data for a scatterplot of estimated vs. true reflectance

irnet/analysis
- Python code to produce Figure 6 (illusion strengths for humans and models) and scatterplots in the paper


3. Classical models

classical
- MATLAB code to apply the models to illusion images
- MATLAB and Python analysis code to find and plot matches for each model

classical/models
- MATLAB implementations of the three classical models evaluated in the paper

classical/scatterplots
- Python code that reproduces the scatterplots shown in the paper for the models


4. Illusion stimuli for the simulated matching experiment

illusions_matching
- illusions for simulated matching tasks for IRNet and the classical models
- provides several images of each illusion, with the reference patch at a fixed reflectance, and the match patch at a range of reflectances
- files with 'mask' in their name provide masks that make it possible to select the reference and match regions of the corresponding illusion images
