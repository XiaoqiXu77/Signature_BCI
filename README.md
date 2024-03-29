# Signature method for brain-computer interfaces

The implementation of two signature-based methods applied on EEG-based brain-computer interfaces (BCIs). The first method uses the path signature directly as a feature vector. The second method takes negative square of the lead matrix constructed from the second level signature and adds a regularization term to get a symmetric positive definite (SPD) matrix to be used as features. 

A simple example of classification of left/right motor imagery of one subject in [Physionet MI dataset](https://physionet.org/content/eegmmidb/1.0.0/) is given. However these methods are general and can be applied to other paradigms of EEG-based BCIs.

![alt text](https://github.com/XiaoqiXu77/Signature_BCI/blob/main/Sig_BCI.png?raw=true)

## Dependencies

* Numpy
* Scikit-learn
* MNE
* PyRiemann
* PyTorch
* Signatory

## Citation
Xu, X., Lee, D., Drougard, N. et al. Signature methods for brain-computer interfaces. Sci Rep 13, 21367 (2023). https://doi.org/10.1038/s41598-023-41326-8

## Contact

If you have any questions or suggestions, feel free to contact via 77xiaoqiqi at gmail.com
