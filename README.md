# LocalChainMotifs_ParadoxicalEffect
 This repository provides the code to reproduce the corresponding figures in the main text of the paper “Identifying the impact of local connectivity patterns on dynamics in excitatory-inhibitory networks”.

Please run the Jupyter notebook files starting with 'Fig**_'. The parameters for each case are specified within the manuscript, and you may need to manually adjust them. Some figures can be generated using a single notebook with the same network parameters (e.g., Fig2_5_6_Gaussian_matrix.ipynb generates Figures 2, 5, and 6).

For clarity, the expected output figures are generated inline within the notebook. You can still run the simulation according to the instructions; however, please note that this may take some time, as the number of trials multiplied by num_taus is large. Code for conveniently saving the data to your local folder is also provided (def list_to_dict()). After saving, you can easily retrieve the data later to regenerate the figures (set RERUN=0).

Please be aware that the results you generate may vary slightly due to the random number generation used in the numerical simulations.

For the generation of true Sparse network with second-order motifs, we uses sonets for reference. sonets is an algorithm to Generates Second Order Networks (SONETs) with prescribed second order motif frequencies. The algorithm is detailed in the following paper:

--L. Zhao, B. Beverlin II, T. Netoff, and D. Q. Nykamp. "Synchronization from second-order network connectivity statistics." Frontiers in Computational Neuroscience, 5:28, 2011.
You can access the source code for SONETs at the following GitHub repository, which is available under the GPL-3.0 license: https://github.com/dqnykamp/sonets#GPL-3.0-1-ov-file.

Pre-generated data that can be re-loaded and used to re-generate the main Figures is in https://www.dropbox.com/scl/fo/jbo2m59tpsvz2mzkpq8q6/AMN-YXiFLY1A872MWzgzsiQ?rlkey=0l2yyw95ycp76ycb83mmh6h5z&st=eyqizsb1&dl=0 .

In case of any questions, requests, or problems, please feel free to contact me (ivyerosion@gmail.com)
