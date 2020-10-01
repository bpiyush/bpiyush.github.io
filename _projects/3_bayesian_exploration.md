---
layout: page
title: Bayesian Exploration
description: Hyperparameter optimisation through Bayesian Inference
img: /assets/img/pmi-1.png
importance: 2
---

Hyperparameters are ubiquitous in machine learning models. Almost every machine learning model has a set of hyperparameters that need to be fine-tuned to get desirable performance of the model. For example, linear regression, arguably the simplest machine learning model has the precision of the likelihood and the prior distributions as the hyperparameters. These meta-parameters become even more critical in neural-network based models. It is often observed that tuning hyperparameters to their optimal values leads to substantial improvement in performance.

In a hyperparameter optimization setting, we are essentially trying to solve an optimization problem with an additional twist: the objective function is known only for a small number of points. Assuming there exists some implicit mapping from the space of hyperparameters to the space of evaluation metric, called as the *black-box function*, our goal is to find the a point in the hyperparameter space that will maximize minimize the performance metric. The approach of automated tuning relies on the inference that we are able to make considering the performance on the past set of hyperparameter configurations and then using this inference to decide the choice of hyperparameters to follow next.
This optimization is at two levels: Firstly, we are trying to achieve a reasonable estimate of the black-box function, and secondly, we are trying to reach to the optima of this blackbox function.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/hyperparam_tuning.png' | relative_url }}" alt="" title="example image"/>
    </div>
</div>
<div class="caption">
    <b>Problem of hyperparameter tuning</b>: We have a set of points in the hyperparameter space $\mathcal{X}$ on which we know the metric value obtained through task $T$. The goal is to obtain that configuration of the hyperparameters which will optimize the metric.
</div>

