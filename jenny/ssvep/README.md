Author: Jenny Qinhua Sun

This directory contains ready-to-be-used data pipeline to feed brain signals from the pdmattention dataset (N200, SSVEP power, raw signals) to FCN (Fully Connecte Neural Networks) and CNNs (1d Convolutional Neural Network)

=======
## Introduction
As humans, we constantly interact with the world by quickly allocating attention, detecting stimuli and making decisions. Electroencephalogram(EEG), compared with other neuroimaging techniques, has high temporal resolution and thus has been widely used to investigate the neural correlates between brain signals and behavioral response. Traditionally, event-related potential (ERP) components has been a well-established measure to study human cognition, as the Signal-to-noise-ratio(SNR) are stronger when time series are obtained by averaging across trials and subjects. Critically, a growing body of research has focused on developing new computational apporaches to extract features from single-trial EEG and link them with behavior during decision making[1]. 

Decision making process is greatly modulated by attention, as it could preferentially faciliate the processing of task-relevant signals[2]. In the field of Brain-Computer Interfaces(BCI), attention could be measured using signals such as N200 (or N1) and Steady-State Visual Evoked Potentional (SSVEP). N200 signals are negative peaks around 200ms post-stimulus that are associated with the onset of visual signal in evidence accumulation. SSVEPs are natural responses to visual stimulus flickering at certain frequencies. More specifically, single-trial SSVEP could be beneficial for patients with severe phsyical disabilities to complete daily tasks by fixating upon the simuli of interest. There has already been research trying to use deep learning neural network on SSVEP for classification task[3].  In the current project, SSVEPs would be the brain responses to 30Hz signals follwed by target stimulus and 40Hz signals followed by distractors. The energy at these two frequencies would accordingly represent the deployment of attention, either to the target or the distractor. The goal of the current project is to train two neural networks on trial-level EEG data, in order to see whether the model(s) could correctly predict accuracy (classfied as correct or incorrect) and reaction time (classfied as fast, medium, slow RT) within that trial. Figure 1 below demontrates the meausre of interest.

![image info](https://github.com/jennyqsun/PSYCH239NNML_Project/blob/main/Figures/demo_n200.png)<br />
**Figure 1.** *N200 signal averging from all trials from one subject*

![image info](https://github.com/jennyqsun/PSYCH239NNML_Project/blob/main/Figures/demo_ssvep.png)<br />
**Figure 2.** *Enhanced SSVEP spectrum summing from all trials using. Top two figures represent the photocells that record the stimulus at each frequency. Bottom figure shows the corresponding power at 30Hz and 40Hz obtained from Fast Fourier Transform (FFT) of the brain signals. As 40Hz has more power than 30Hz, we would interpret it as the subject deployed more attention to the distractor flicker, relative to the target flicker.* 


## Methods
### Task 
The current Task is a S-R Mapping Task. The subjects were asked to discriminate 2.45 cpd Gabor from 2.55cpd Gabor. There were three difficulty levels, with the easiest being responding to either stimulus, the medium one being responding with one of the two buttons based on the spatial frequency, and the hardest one being responding to the spatial frequency contingent on a certain orientation. However, considering the current goal of the project and the size of the dataset, we will treat all trials equally. The task had a cue interval followed by a response interval (see Figure 3). During the cue interval, two SSVEP noise signals flickered on both visual fields at two different frequencies, 30Hz and 40Hz for 0.5s - 1s as a probablistic cue. During the response interval, the target stimulus occured on either side of the visual field flickering at 20Hz, with a 30Hz noise in the background (see Figure 4). Accuracy and reaction time were collected and will be used as the class for the classification task.

![header image](https://github.com/jennyqsun/PSYCH239NNML_Project/blob/main/Figures/demo_task.png)<br />
**Figure 3.** *Timeline of the Task.*<br />

![image info](https://github.com/jennyqsun/PSYCH239NNML_Project/blob/main/Figures/demo_signalnoise.png)<br />
**Figure 4.** *How the actualy noise and target stimulus looke like.*<br />


### Dataset Structure and Data Preprocessing
The dataset contained 46 subjects data in total. 12 subjects were randomly selected to be used as testing set, and 33 subjects will be used as training set as one subject was excluded due to low accuracy. Within each subject there are 360 trials in total. EEG data was collected using High Density EEG cap with 128 channels. All the EEG data was cleaned by performing Independent Component Analysis (ICA) and had the obvious non-brain signals removed. A trial is identified as a goodtrial when the artifact from all 128 channels didn't reach certain threshold, and when the RT is over 300ms. Within each subject, RT was broken into tertiles and labeld as slow, medium and fast RT before the trials were stacked.The input shape of the training set then become 33 subjects x goodtrials x features = 11134 trials x features. The final Test sets has an input shape of 12 subjects x goodtrials x features = 4038 trials x features. 

#### Feature Selection
The features for each sample could be broke down to two categories: Power for each channel at certain frequencies, Time Series data with the N200 time window, and N200 parameters (latency and amplitude). The chart below shows the general framework of the input data. Table 1 below shows the feautures that were explored.

![image info](https://github.com/jennyqsun/PSYCH239NNML_Project/blob/main/Figures/input_chart.png)<br />
**Table 1.** *The inputs that were fed to the neural networks. When combining Power and N200 features for CNN, the 2x1 vector for N200 signals are treated as an extra channel.*<br />

#### Oversampling
It's worth mentioning that 0.79% of the train set trials are correct trials, and 73% of the test set trials are correct trials. In order to avoid bias in the train set, incorrect trials were oversampled to match up with the correct trials. This makes the final number of training sameples become 17564 trials, and the testing samples stayed the same.

#### Single Trial Signal Estimate
Since single trial EEG signals could be very noise, and we plan on using each channel as a feature, we will use an algorithm based on Singular Value Decomposition[2]. Using SVD we will be able to identify an optimal set of weights for all channels to get better SNR at frequencies of interest or N200 amplitude. In the frequency domain, we perform a FFT on the ERP for each trial, and pass the fourier coefficients at target frequency and its neighouring frequencies to SVD. The V* matrix then contains an optimal weights for each channel such that it would maximize the power at frequency of interest. We then multiply the S Matrix back as V* contain eigenvalues (unit vector). In the time domain, the method is similar except that we passed a certain time window and get a set of weights to capture the most variance for N200 signals. Fiure 5 (left) demontrates the SVD procedure[4].

#### Domain Adaptation
When combining all the trial across subjects, we assumed that there would be no individual differences among subjects. And that might hurt the performance of the models. Therefore, we used a correlation alighnment algorithm to normalize the features of each such subject based on one specific subject we picked, such that all we transform all subjects' features to one domain[5]. This was implemented by using by Python Package "transfertools." Figure 5 (right) demonstrates how the unsupervised transfer learning technique was applied by using the first and second order statistics of the source and target data for domain adpatation.

![image info](https://github.com/jennyqsun/PSYCH239NNML_Project/blob/main/Figures/Algorithms.png)<br />
**Figure 5.** *Left: SVD procedure in 2D space. In our data it would be 121 channel dimensions. Right: Domain Adpatation.*<br />


### The two neural networks 
#### Fully Connected Neural Network
The first neural netowrk was a simple two-layer fully connected network (see Figure 6). Dropout rate was set to be .5 to prevent from overfitting. The loss function was calculated by cross entropy, and the optimizor was Adam using a learning rate at 1e-3. We used fully connected neural network here because there would be no assumption that there is structure within each EEG channel. When we added the N200 parameter features, the input layer would be [1, 244]. When the output is to classify RT, there are 3 classes instead of 2.

![image info](https://github.com/jennyqsun/PSYCH239NNML_Project/blob/main/Figures/fcn.png)<br />
**Figure 6.** *Two-layer fully connect neural network for 121 30Hz channels and 121 40 Hz channels*<br />
Figure source: https://towardsdatascience.com/coding-neural-network-forward-propagation-and-backpropagtion-ccf8cf369f76

#### Convolutional Neural Networks
The second neural network was a convolutional neural network using conv1D (see Fiugure 7). The general architecture was adpated and modified from a pervious CNN model used for SSVEP classification task[3]. The network is composed of three layers, two convolutional layers and an output layer. One maxpool was implemented after C2, following by a dropout layer and a flatten step.

![image info](https://github.com/jennyqsun/PSYCH239NNML_Project/blob/main/Figures/cnn1d.png)<br />
**Figure 7.** *Two-layer CNN neural 121 30Hz channels and 121 40 Hz channels*<br />
Figure adapted from [3]

## Results
#### SSVEP
When feeding 242 features containing 30Hz and 40Hz power across 121 channels to the two-layer fully connected neural network without oversampling, the training was able to converge closer to 1, but the model doesn't seem to learn much useful information. Either oversampling or not doesn't seem to make a difference (Figure 8a, 8b)).

It's worth mentioning that given that 73% of the trials in the testing set are from one category, if the model learned absolutely no information, it will make all predictions as one category and thus reach 73% accuracy. However, deviating from 73% doesn not necessarily mean that it picked up any useful feature. Therefore, a precision score is added to measure performance, calculated by *true positive / (true positive + false positive)*. The idea is that if the precision score remained the same, even if the model picked up some strucutres, it's likely not valid. In order to error check the model, direct labels representing the each trial's category was added to the data, and the model was performing perfectly (Figure 8c).
![image info](https://github.com/jennyqsun/PSYCH239NNML_Project/blob/main/Figures/fcn_ssvep242.png)<br />
**Figure 8.** *FCN with 242 features*<br />

The two N200 features were also added to the input features. The N200 peak and latency were two relative measures obtained by subtracting ERP peak and latency enhanced by the SVD procedure. Figure 9 demontrated that the training accuracy was able to slowly converge, testing accuracy stayed low (.68). The precision score stayed unchanged.
![image info](https://github.com/jennyqsun/PSYCH239NNML_Project/blob/main/Figures/fcn_ssvep244.png)<br />
**Figure 9.** *FCN with 244 features*<br />


We then seperately fed them into CNN using CONV1D. Input shapes were either fed as a 1D vector [1 channel x 242 or 244 features, or 2 channels x 121 or 122 features], depending on whether including N200 parameters or not. None of the model seemed to extract useful information. None of the model seemed to extract useful information from the input (see Figure 10). However, it is worth mentioning that the testing accuracy from SSVEP + N200 feature as 1 channel produced the highest accuracy that became closer to precision (upper right in Figure 10) 

![image info](https://github.com/jennyqsun/PSYCH239NNML_Project/blob/main/Figures/conv1d_ssvepn200.png)<br />
**Figure 10.** *CNN with 242 or 244 features, treating as 1 or 2 input channels.*<br />



#### N200 Time Series
Lastly we used N200 times series data by channel to the CNN. The input shape was [119 x 500], meaning that there were 119 good channels and 500 time steps. The 500 second window was picked as the first 500ms after the onset of the stimulus. None of the model helps improve the performance on test set (see Figure 11). The channel by time steps input on the left was non-domain adapted time series, and it seems like even the training set learned no information at all. The model made guesses as all accurate or all inaccurate and got either 0.73 or 0.27 accuracy score.


![image info](https://github.com/jennyqsun/PSYCH239NNML_Project/blob/main/Figures/cnn_n200time.png)<br />
**Figure 11.** *CNN using either EEG single channel by time steps, or across channels by time steps.*<br />


#### Classfying Fast/Medium/Slow RT
None of the above approaches seemed to have any predicting power on RT. Figure is an example of the null results, and therefore will not be presented seperately for each model. 
![image info](https://github.com/jennyqsun/PSYCH239NNML_Project/blob/main/Figures/rtclass.png)<br />
**Figure 9.** *242 features used on classifying three classes of RT.*<br />




## Summary and Future Direction
This project aims to use different EEG modalities to classify correct and incorrect trials, and to classify whether RT belongs to one of the three conditions. However, none of the features either alone or jointly succussfully helped the model learn. Two conclusions may seem to be useful. First, it seems that at least SSVEP features and N200 features jointly could be more trainable when evaluating with the training set, relatively to N200 time series. Perhaps these features worth further exploring as they may share some features that could be extracted. It's possiblity that raw time series, even with preprocessing, are still too noisy compared to power spectram that were extracted. Second, it's unclear whether the domain adaptation method works for EEG data as obtained conflicting results when comparing with raw data versus domain adapted data. It's possible that some other normalization would be needed for these EEG data.

There are sevel future directions: 1) Using trial-level spectrogram (or Hilbert Huang Transform results). The fact that we only used two key frequencies are their spectral power for a time window might not be optimal to portray the full profile of each trial. 2) Modifying the CNN as a regression and use only the accurate trials. Accurate trials could be due to certain features, but inaccurate trials could be due to a mixture of factors. Therefore, using the model to learn the strucutre and classifying the two categories might not be ideal. Using all the features and regress on the RT may be a better approach. Perhaps also combine accuracy and RT as one measure. 3) Picking some of the channels may reduce the noise. In the SSVEP and ML literature, people usually pick channels over visual cortex to reduce noise[3]. 4) Picking a separate testing and training set. 5) Using other behavioral parameters by trial-blocks, and link them with the blocked EEG signals too to improve SNR.

# References 
1. Bridwell, David A., James F. Cavanagh, Anne G. E. Collins, Michael D. Nunez, Ramesh Srinivasan, Sebastian Stober, and Vince D. Calhoun. 2018. “Moving Beyond ERP Components: A Selective Review of Approaches to Integrate EEG and Behavior.” Frontiers in Human Neuroscience 12. https://doi.org/10.3389/fnhum.2018.00106.

2. Nunez, Michael D., Joachim Vandekerckhove, and Ramesh Srinivasan. 2017. “How Attention Influences Perceptual Decision Making: Single-Trial EEG Correlates of Drift-Diffusion Model Parameters.” Journal of Mathematical Psychology 76 (Pt B): 117–30. https://doi.org/10.1016/j.jmp.2016.03.003.

3. Kwak, No-Sang, Klaus-Robert Müller, and Seong-Whan Lee. 2017. “A Convolutional Neural Network for Steady State Visual Evoked Potential Classification under Ambulatory Environment.” PLoS ONE 12 (2). https://doi.org/10.1371/journal.pone.0172578.

4. “Singular Value Decomposition.” 2020. In Wikipedia. https://en.wikipedia.org/w/index.php?title=Singular_value_decomposition&oldid=993831805.

5. Sun, Baochen, Jiashi Feng, and Kate Saenko. 2015. “Return of Frustratingly Easy Domain Adaptation.” ArXiv:1511.05547 [Cs], December. http://arxiv.org/abs/1511.05547.

