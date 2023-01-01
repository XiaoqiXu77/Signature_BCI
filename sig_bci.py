"""
Path signature applied on electroencephalography-based brain-computer interface
"""

import torch
import signatory
import numpy as np
from mne import Epochs, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from pyriemann.classification import TSclassifier
from pyriemann.estimation import Covariances
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def read_data(subject, runs, event_id):
    '''
    return epoched and filtered EEG data and event labels

    Parameters
    ----------
    subject : int (positive)
        subject id number.
    runs : list
        experiment runs concerning the events to be extracted.
    event_id : dictionary
        defining labels for the events.

    (see https://physionet.org/content/eegmmidb/1.0.0/ for more details on runs and event_id)
    
    Returns
    -------
    epoches_data : ndarray, shape (n_epochs, n_channels, n_samples)
        epoched and filtered EEG data.
    labels: ndarray, shape (n_epochs,)
        event labels of all epochs
    '''
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])

    # Extract information from the raw file
    events, _ = events_from_annotations(raw, event_id=dict(T1=0, T2=1))
    raw.pick_types(meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    # preprocessing
    raw.filter(8, 30, fir_design='firwin')

    # Epoching and load data
    tmin, tmax = 1., 3.
    epochs = Epochs(raw, events, event_id, baseline=None, tmin=tmin, tmax=tmax)
    epochs.drop_bad()
    labels = epochs.events[:, -1]
    epochs_data = epochs.get_data() * 1e6
    return epochs_data, labels

def first_method(data, depth):
    '''
    return the path signature truncated at level 'depth' as feature vector

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_samples)
        EEG data.
    depth : int
        truncation depth of the path signature.

    Returns
    -------
    feature : ndarray, shape (n_epochs, \sum_{d \le depth} C_{n_channels}^d)
            truncated path signature as feature vector.

    '''
    data_sig = np.swapaxes(data, 1, 2)
    path = torch.from_numpy(data_sig)
    signature = signatory.signature(path, depth)
    feature = signature.numpy()
    return feature

def second_method(data, depth, epsilon):
    '''
    return signature-based SPD matrices as features

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_samples)
        EEG data.
    depth : int
        truncation depth of the path signature.
    epsilon : float
        regularization parameter.

    Returns
    -------
    feature : ndarray, shape (n_epochs, n_channels, n_channels).
        signature-based SPD matrices as features.

    '''
    n_chan = data.shape[1]
    data_sig = np.swapaxes(data, 1, 2)
    path = torch.from_numpy(data_sig)
    signature = signatory.signature(path, depth)
    feature = signature.numpy()
    
    # lead matrix
    sig2 = feature[:, n_chan:].reshape(-1, n_chan, n_chan)
    L = sig2 - np.swapaxes(sig2, 1, 2)
    A = - np.matmul(L, L)
    feature = A + epsilon * np.identity(n_chan)
    return feature

def main():
    '''
    A simple example of classification of left/right motor imagery of 
    one subject in Physionet MI dataset using two signature-based methods

    '''
    # read EEG data
    subject = 7
    runs = [4, 8, 12]  # moter imagery: left vs right
    event_id = dict(left=0, right=1)
    data, labels = read_data(subject, runs, event_id)
    
    # truncation depth of signature
    depth = 2
    
    ### first method ###
    feature = first_method(data, depth)
    
    # classification 
    sca = StandardScaler()
    lr = LogisticRegression()
    ppl = Pipeline([('scaler', sca), ('clf', lr)])
    n_fold = 10
    cv = KFold(n_fold, shuffle=False)
    scores_1 = cross_val_score(ppl,
                               feature,
                               labels,
                               cv=cv,
                               n_jobs=-1)
    
    ### second method ###
    epsilon = 0.001  # regularization parameter
    feature = second_method(data, depth, epsilon)
    tsc = TSclassifier()  # Riemannian classifier
    scores_2 = cross_val_score(tsc, X=feature, y=labels, cv=cv)
    
    print("Mean accuracy of the first method: {0:4.3f}".format(np.mean(scores_1)))
    print("Mean accuracy of the second method: {0:4.3f}".format(np.mean(scores_2)))
    return