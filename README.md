# AMLR
Code release for Adaptive Multi-Scale Language Reinforcement for Multimodal Named Entity Recognition

# Datasets
Please follow https://github.com/jefferyYu/UMT to get the image data.
The textual data is located in the "data" folder, within the "twitter2015" and "twitter2017" subfolders.
Please confirm the data path and parameters before training or testing.
# Environment
Python==3.8, torch==1.3.1, transformers==3.0.0 

# How to run!
For Twitter 2015, please run 'sh joint_method_15.sh'  or 'python run_joint_span.py --params <param_values>'(where "params" are the argument names of the parser in run_joint_span.py, "param_vales" are the value of "params") for training our model. When training, set "do_train" to "True" and "do_predict" to "False". The same applies to Twitter 2017.

