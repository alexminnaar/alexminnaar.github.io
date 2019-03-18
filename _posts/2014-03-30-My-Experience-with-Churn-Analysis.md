---
layout: post
title: "My Experience with Churn Analysis"
date: 2014-03-30
comments: false
categories: 
---

A large chunk of my time at my last job was devoted to churn analysis and I wanted to use this blog entry to explain how I approached the various problems that it presented.  This is not meant to be a very technical post and the reasoning behind this is two-fold

1. Obviously I do not have permission to use any company data and there is not really any good publicly-available churn datasets on the web.  Presenting technical code without data to run it on would not really make sense.
2. I have learned that churn analysis is very domain-specific and I want to make sure that what I say generalizes to many use-cases.

Before I explain what I did, I should first define what churn is and the specific goals that I had in mind.

## What is Churn?

Churn is a term that generally describes the process where customers stop using the products and/or services provided by a business.  However, it is of most interest in subscription-based services like phone plans, video games, etc.  In these services it is easy to know when a customer has churned i.e. when they cancel their subscription.  Needless to say, churn is bad for business.  Every company has a customer CPA (Cost Per Acquisition) so in order to replace a churned customer with a new customer, this cost must be paid.   Clearly it is cheaper for companies to keep customers than to replace them.  Churn analysis is used to attempt to answer the following questions

1. Why are customers churning?
2. Can we predict customers who will churn in the near future (and then maybe convince them to stay with us)?
3. How long can we expect a given customer to stay with us?

These are very important questions and if reliable answers can be obtained, they would be of great value.  We will also see that these main questions are closely linked to some other slightly different yet equally important questions such as

1. What is the lifetime value of a given customer?
2. Who are our most valuable customers and who are out least valuable customers? What accounts for these differences?

## Question:  Can we predict which customers will churn in the near future?

Predicting which of your current customers will churn is a binary classification problem.  As it stands, this is an ill-defined problem.  This is because of the simple fact that in subscription-based services ALL CUSTOMERS WILL CANCEL EVENTUALLY!  You could have a classifier that predicts that all currently active accounts will cancel and it would be 100% correct!  But obviously this would be useless to a company.  What companies really want to know is which of the currently active accounts will cancel "soon".  This way companies can take action in an effort to prevent cancellation from occurring.  The specific preventative action that should be taken is beyond the scope of this blog post but the prediction problem itself will be explored.

## Dealing with the Time Component

The first thing that you need to do is define a time period.  There is a trade-off here.  You want to know who is going to cancel as soon as possible so that you have the maximum amount of time to take preventative action. However if you predict too early, your predictions will be of lower quality.  This is because (in most cases) churn indicators become clearer the closer the customer is to his/her actual cancel date.  On the other hand, if you predict too late, your predictions will be more reliable but it will give you less time to take preventative action.  You need to decide on a good balance which most likely depends on your domain.  I can say that in the telecom domain a 2 week window is generally enough time to perform preventative action.

Once you have dealt with the time component, the classification problem becomes more well-defined however there is still a bit more work that needs to be done.

## Defining Positive and Negative Examples

In any classification problem you need to build a training set of positive and negative examples.  It is clear that negative examples will come from customers that have churned in the past.  However it is a bit unclear what the positive examples should be.  You might initially think that we can use the currently active accounts as positive examples.  This is problematic sinse ultimately these are the accounts we will test on so we can't really use them for training as they are.

What you need to do is identify your long-time customers (they will most likely be currently active, but they could also have cancelled after using your service for a long time).  However, as previously stated, you cannot use them as they are because you are going to test on them.  You need to use the truncated versions of these examples as positive examples.  For example, if you have a long-time customer that has been active for two years, use this customer's behaviour from their first 365 days as a positive example.  In this way, you obtain positive examples of customers that you know will not cancel for a long time.  Also, testing will generally be done on an active customer's recent behaviour, so you are mitigating the risk of overfitting by training on that customer's past behaviour.

Now that you know what your positive and negative examples are you must extract relevant feature from them.

## Feature Extraction

The feature extraction process is the most important part of this problem and, unfortunately, also the most unsystematic.  If you have dealt with supervised learning problems before you know that feature extraction is as much an art as it is a science.  Feature extraction is very domain-specific.  In some cases the relevant features that indicate churn likelihood are obvious, in others it is less clear.  It would be wise to consult someone with good domain expertise before you decide on the features you will use.  I will list some of the features that I found to be good indicators of churn in the telecom domain.

## Static Features

Static features are features that are not time-dependent.

* Age at activation date
* Lead source
* Type of phone
* Number of phones attached to account
* Location
* Credit card type

## Usage-based Features

Usage-based features deal with the customer's time-dependent usage patterns.

* Date of last usage
* Max and min daily usage amount
* Average usage amount over last 30 days
* Average usage amount over last 30 days / overall average
* Number of support tickets issued
* Number support tickets in last 30 days / total # of support tickets
* Max # of days without any usage
* Current # of days without any usage / max # days without any usage

However, as I said earlier, feature extraction is a very domain-specific problem so there is no guarantee that these features will be useful in your particular use case.

## The Class Imbalance Problem

In almost all applications of this problem you will find that you have many more active accounts than cancelled accounts.  Therefore you will have many more positive training examples than negative training examples.  This is problematic because any classifier that predicts that no customers will churn will perform very well.  Consider the case where you apply the classifier to a set of 100 accounts - 90 that will not cancel in the next 2 weeks and 10 that will.  If the classifier predicts that all 100 will not cancel, it would have an accuracy of 90%.  Even though the classifier is very accurate, it is of little use because we need to identify these 10 accounts that are going to cancel.  This is called the class imbalance problem.

There has been a fair amount of research into this problem and survey of possible solutions can be found [here](http://marmota.dlsi.uji.es/WebBIB/papers/2007/1_GarciaTamida2007.pdf).  I have found that over-sampling the negative examples works well.  Specifically, I used stratified sampling on the negative examples such that my final training set contains a certain percentage of negative examples.  There is another trade-off here.  The higher the percentage of negative examples, the more false negatives (incorrectly predicting that a customer will churn) you will generate.  But if the percentage is too low, you will miss accounts that will cancel.  You must decide the threshold of false negatives that you can tolerate.

## Putting it All Together

Now that you have defined your positive and negative examples, extracted features and dealt with the class imbalance problem, you can finally build your model.  The particular model that you choose is up to you.  I have found that random forests perform well in most applications.  The best results generally come from an ensemble of multiple models.  Obviously you would want to perform the usual train/test set splitting, cross-validation and parameter tuning that is required to reduce overfitting.  Once your model is trained up to your standards, you will apply it to your set of currently active customers.

You also want to decide how often to run this classifier.  Since the usage-based features are constantly changing, running the classifier frequently would be a good idea.  However, if you notice that no new accounts are being flagged, it might be a good idea to run it less frequently.  But if you have the capacity, there is really no downside to running it as often as possible.