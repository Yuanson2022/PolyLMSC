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
"Data" folder (Wait update):
Note: The data are being organized to achieve better reproducibility, We will upload them upon acceptance.

	-Training texture
 		# Each training texture originall contains 10,000 rough paths and 10,000 smooth paths
		# We filter out some abnormal paths, so the actual path number has a slight difference
		1. Random texture  xx.hdf5
		2. Cube texture    xx.hdf5			

	-Mixed texture for generalization validation
   		# Each texture contains 400 rough paths and 400 smooth paths
		# Purely used for validation
		1. Mixed texture 1  xx.hdf5
		2. Mixed texture 2  xx.hdf5
		3. Mixed texture 3  xx.hdf5

Content details:

	-Mechanical sequence is in the shape of [path_num, seqential_length, featrue_dimension].
 
	-Texture sequence is in the shape of [path_num, grain_number, seqential_length, featrue_dimension].

  	-Initial textures and associated Fourier coefficients are provided.

We also provide code to visualization this path...
