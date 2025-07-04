# Code and data for the PolyLMSC model
#  --------------------------------------------------------
PolyLMSC is a mean-field neural network-based crystal plasticity model.The dataset and codes, involved in the model development, are attached to the article:
"Multiscale modelling with neural network-based crystal plasticity model from meso- to macroscale"

If you use the dataset or codes, please cite them.
#  --------------------------------------------------------
Author: Yuanzhe Hu

Affiliation: State Key Laboratory of Mechanical Systems and Vibration, Shanghai Jiao Tong University
#  --------------------------------------------------------
"Code" folder:

LMSC_model.py:

	-Stress LMSCs
		1. x: Input strain increment to the polycrytalline aggregate [batch, seq, feature_dim]
		2. init_F: Fourier coefficent of initial texture [296,1] 
		3. out_state_list: Predicted stress components [batch, seq, feature_dim]
		4. alpha_out: Evoluation rate of stress states for the robustness fine-tuning

  	-Texture LMSCs
		1. x: Input strain increment
		2. init_ori: Initial orientations of the compact set [100,3]
  		3. out_state_list: Predicted texture in Euler angles [batch, grain_num, seq, feature_dim]
		4. alpha_out: Evoluation rate of texture states for the robustness fine-tuning
#  --------------------------------------------------------
Data availability:

The complete dataset is available at https://zenodo.org/records/15742872

Content Details:

-The repository contains two .rar files, corresponding to the mechanical sequence and texture sequence used in the paper.

-Mechanical Sequence: (1) Stored under the 'data' group in HDF5 files; (2) Shape: [path_num, sequence_length, feature_dimension], with sequence_length = 2500; Feature dimension includes 12 variables:[L11, L12, L21, L22, ε11, ε22, ε33, ε12, σ11, σ22, σ33, σ12].

-Texture Sequence: (1) Stored under the 'tex' group in HDF5 files;(2) Shape: [path_num, grain_number, sequence_length, feature_dimension], with sequence_length = 101, and feature_dimension = 3 (Euler angles).

-Training texture (Random texture / Cube texture): Each training texture originall contains around 10,000 rough paths and 10,000 smooth paths; We filter out some abnormal paths, so the actual path number has a slight difference.
		
-Mixed textures for generalization validation (Mixed texture 1/2/3): Each texture contains 400 rough paths and 400 smooth paths; Purely used for validation.

-The Fourier coefficients of initial texture are provided in each sub-path of 'stress-data'.
 
-A script for visualizing individual loading paths is provided in the 'stress-data.rar'.

