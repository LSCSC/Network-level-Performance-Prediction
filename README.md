# Network-level-Performance-Prediction

This repository saves all source codes for the conference paper *Network-Level System Performance Prediction Using Deep Neural Networks with Cross-Layer Information* on 2020 IEEE International Conference on Communications (ICC), and another journal paper to be published. The dataset is stored [elsewhere](https://pan.baidu.com/s/1SeVaT4e0YPyx6rVbrHjNJg) (due to the limited space of github) with the file accessing code **cucq**.

The folder Python_Source_Code keeps all the source codes for the machine learning algorithm in the paper, and the Matlab_Code contains all the codes for plotting the resulting figures.

For the machine learnig related codes:
- *main_ack.py* trains ACKNet to predict the ACK/NACK outcomes.
- *main_uat_mape_coteach_cl.py* trains UATNet to predict the user average throughput (UAT) for a target user.
- *model_set_jf.py* includes the DNN configurations.
- *TrainTest.py* collects some auxilary functions.
- *mcs_landscape_ack.py* and mcs_landscape_uat.py plot the MCS landscapes in the papers from ACKNet and UATNet.
- *get_confusion_matrix.py* calculates the confusion matrix for the classification problem (ACK/NACK prediction).
