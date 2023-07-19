# Precise Diagnosis of Parkinson's Disease using an Ensemble of Neural Networks
A Regression approach for the automated detection of Parkinson's Disease based on an Ensemble of Neural Networks.
Note that this is still a Work in Progress, and numerous modifications and updates will follow.

</br>

![alt text](https://github.com/shahriar-rahman/Precise-Diagnosis-of-Parksinsons-Disease/blob/main/img/parkinsons1.jpg)

</br>

## ◘ Background
Parkinson’s disease (PD) is one of the most ubiquitous progressive and chronic neurodegenerative disarrays caused due to the deterioration of substantia nigra, a collection of predominant nerve cells which operate with neurotransmitters responsible for regulating and coordinating muscle cells involved in a myriad of body movements. Due to the impairment of such cells, the production of dopamine is critically ceased, resulting in an unbalanced aggregation of the alpha-synuclein protein in the glial cells, neurons, and nerve fibers, therefore, impacting the quality of life as social interactions become more challenging and further depreciate the financial state of the patient with an enormous expense for incremental medical healthcare costs. Currently, there is no cure for PD and its primary cause remains a mystery albeit a minor percentage can be attributed to an amalgamation of researched genetic influences and environmental factors such as fungicides, herbicides, and toxins. According to a study conducted in 2016, the number of individuals associated with PD reached around 6.1 million, compared to 2.5 million in 1990, and instigated well over 3.1 million deaths, revealing a massive surge in the prevalence worldwide as its extended nature and the demographics affected, especially the late 50s and onwards, are poised to substantially intensify and exasperate the burden of Parkinson’s disease in the future.

</br>

## ◘ Distribution Visualization
![alt text](https://github.com/shahriar-rahman/Precise-Diagnosis-of-Parksinsons-Disease/blob/main/Diagrams/PD-Distribution.PNG)

</br>

## ◘ Objectives
The primary incentive of this paper is to introduce an automated PD diagnosis framework that incorporates several motor features such as acoustic speech, rest tremors, clinical characteristics, bradykinesia, and hypokinesia from patients diagnosed with PD, healthy controls (HC), and patients with RBD. Many healthcare systems have adopted various rudimentary Machine Learning (ML) techniques to accurately classify PD. However, contrary to popular belief, there are multitudinous advantages that Neural Networks possess over other prevalent ML algorithms such as examining data with a logical structure parallel to how humans comprehend, flexibility, the ability to make self-adjustments, and robust generalization with complex non-linear, and high-dimensional data. 

</br>

## ◘ 3D-Diagram of Feature Relationship
![alt text](https://github.com/shahriar-rahman/Precise-Diagnosis-of-Parksinsons-Disease/blob/main/Diagrams/3d_diagrams.JPG)

</br>

## ◘ Methodology
The primary initiative of this paper is to develop a robust intelligence system for the diagnosis of PD by means of multifarious Neural Networks based on the classes of ensemble methods. Each network is devised for the bootstrap aggregation and stacked generalization techniques, while a forward-stage boosting algorithm is applied for boosting system for comparative analysis. All samples of data are amalgamated with a Gaussian synthetic sampling technique to elevate the scope of the sample followed by processing using an attribute selection procedure incorporating an estimator model for computing the feature importance. The designated significant ranking attributes are selected for the regression analysis of the total Parkinson’s scale, or Total-UPDRS, in the aforementioned models, and the universal generalizability of the standalone models as well as their involvement in the ensemble approach are dissected and assessed to evaluate the most effective predictive model or approach with minimum discrepancies for the diagnosis of PD. 

## ◘ Data Scaling
Upon deeper introspection, the range and the scale of the attributes were ubiquitous. Case in point from Table 2, the Respiratory Exchange Latency can diverge from 90 to more than 115 milliseconds, forming a conspicuous pattern that is also indistinguishable from the Rate of Speech as it oscillates at a higher magnitude ranging from 70 to over 100 speeches per minute. On the contrary, the Hoehn and Yahr scale is observed to have trivial aberrations ranging from 0.1 to 0.5 scale. Therefore, the feature information is scaled using the Standardization technique, readjusting with a statistical function presented in equation 2 to harmonize the data across the models, especially Neural Networks, as the values get converged around the mean with a unit of standard deviation.
```
z =  (x-µ)/(σ )                                 
```                                                                            
Where µ is referred to as the mean of the given distribution, σ for its standard deviation, and x for the sample value. Z is representing the quantitative value for standard deviation fluctuation around the mean that a specific observation falls. 

</br></br>

## ◘ Data Partition     
The allocation strategy of the data is based on the Pareto Principle which states that 20% of the vital few transpire roughly 80% of consequences. Therefore, according to the principle of factor sparsity, 80% of the total information is employed for the training phase and the rest 20% for the testing phase. During the partition procedure, all the statistical values are reshuffled both before processing and after processing, averting any potentiality to form any miniature bias by the supervised models through learning the samples.

</br>

## ◘ Flowchart for the proposed System
![alt text](https://github.com/shahriar-rahman/Precise-Diagnosis-of-Parksinsons-Disease/blob/main/img/FlowChart.png)

</br>


