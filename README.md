# BELLA

A deterministic method to explain predictions of black-box regressors. It provides accurate, general, simple and
verifiable explanations.

BELLA computes the optimal neighbourhood around the data point being investigated and then trains a linear regression 
model. The linear regression model represents the explanation for a given prediction.
Given that it has been trained in input feature space, this linear model i.e. explanation is verifiable.
User can manually replace feature values to verify the predicted value. At the same time they can see how changes in 
feature values affect the outcome.

Finally, the explanation can be applied to all data points in the optimal 
neighbourhood, which should increase users' trust in the model and explanation.