# Project Summary
After training a learner and expert according to bc_code.md, we want to anlyze its performance. The first step is to to visualize the train/test loss and accuracy. We use the test accuracy or loss to compute the CL Metrics. The training code saves the weights and representations for every log_freq epoch out of all the training epochs for each task. These are both used in Plasticine analysis. Finally, glue uses the representations to calculate capacity, dim and radius for the class manifolds.

We want to call each of these in the single_run.py script. Theses analysis methods will read in the data saved from the main training script following the format found in bc_code.md. Each analysis will calculate its correpsonding metrics. The results for each analysis will be saved in a way so as to be used in a comparison plotter, where we can compare these metrics between algorithms. Right now We have only RL and SL but will add others in the future. 


### CL Metrics


### Plasticine



### GLUE Analysis

