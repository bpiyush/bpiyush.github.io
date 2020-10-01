---
layout: page
title: Attention-based Zero-shot Classification
description: Learnt attention prior for ZSL tasks
img: /assets/img/zsl.png
importance: 1
---

Humans have a remarkable ability of recognizing object classes even if they have not been exposed to images of such objects. Computer vision systems tend to perform extremely well with sufficient supervision. Typically, the set of classes at test time for such systems are a subset of the classes that constitute the training data. In a zero-shot learning setup, we measure the model's ability to generalize well to unseen classes during test time. For instance, as shown in the following figure, you might have classes like *Tiger*, *Buffallo*, *Horse* etc. in your train set but might encounter a completely different class of wild-animal during inference. Along with images, we also have some form of *class description*, often called as `class attribute vectors` as part of the training set. Henceforth, I will refer to this classification task as ZSL problem.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/zsl.png' | relative_url }}" alt="" title="example image"/>
    </div>
</div>
<div class="caption">
    <b>Zero-shot Learning</b>: We want the model to be able to generalize well for unseen class images at test-time.
</div>


#### Attention for discriminative power

The main challenge in ZSL problem is to make the model align the visual features of an image semantically with the class attribute vectors. However, using global features may prove to be sub-optimal since local features may be more helpful in discriminating closely related classes, some of which may be unseen. The primary motivation of incorporating attention in ZSL setting is to make the model focus and learn more discriminative visual features for better generalization. This can be illustrated by the following example: If we have the class of horses in the training set and that of zebras as unseen class, it may not be beneficial to have local features comprising of description of the feet of the animals. Rather, it may be beneficial to only selectively attend to the local features that are good enough to discriminate between a horse and a zebra. With this motivation, we propose a learnt attention-based prior to alleviate model’s alienity of unseen classes.


#### Approach

We follow the route of embedding the attribute vectors into the visual features space based on the observations in [9]. The class distributions to be learnt in the visual space are learnt based on attention-weighed local region-based visual features. As illustrated in the following figure, for input image, the CNN model provides visual features $\textbf{x}$ which is further passed to the attention network which gives attention-weighted features $\hat{\textbf{x}} = \textbf{x} \odot NN(\textbf{x}, \textbf{a}_{c})$. On the other end, attribute vector $\textbf{a}_{c}$ is used to compute the means and variance of all classes to produce class probabilities for $\textbf{x}$. The combination of the two quantities computes gives the softmax loss which can be minimized using gradient-descent to find optimal settings of neural networks. You can checkout the details of training and inference routines in the [report](https://drive.google.com/file/d/1OkahHv54mfyazKJqItNvR2B2cUdsuh-Q/view?usp=sharing).

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/block-diagram-v2.png' | relative_url }}" alt="" title="example image"/>
    </div>
</div>
<div class="caption">
    <b>Attention in ZSL</b>: CNN-based features are weighed by output of the attention network.
</div>

#### Experimental Evaluation

**Datasets**: We evaluate on the Caltech-UCSD Birds 200(CUB-200) and Animal With Attributes(AWA) datasets. These are standard datasets for ZSL tasks and you can find more details in the [report](https://drive.google.com/file/d/1OkahHv54mfyazKJqItNvR2B2cUdsuh-Q/view?usp=sharing).

**Results**: The following table notes the test accuracies obtained on each of the two datasets. The train-test split is the same
that is used in the literature for ZSL. We have reported top 1 accuracy i.e., average of the accuracy over all the unseen classes. Though we have put the attention module on the global features coming from the ResNet 101 network. it’s still beating the state of the art work on AWA dataset. The model will improve further on incorporating attention over the local features. Other approach to the same could be to have a network instead of ResNet 101 which is used in task like object detection since it will give better features correspondiong to the attention task similar to what is done in Stacked Semantic Attention Framework.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/data.png' | relative_url }}" alt="" title="example image"/>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/results.png' | relative_url }}" alt="" title="example image"/>
    </div>
</div>
<div class="caption">
    [Left] Sample images from the CUB-200 and AwA datasets respectively. [Right] Comparison of top-1 accuracy over all unseen classes in the test set.
</div>

#### Conclusion and Future Work

The proposed approach is one of the early attempts at incorporating attention for Zero-shot Learning. [5] is to the best of our knowledge, the only other attempt at incorporating attention in ZSL. Our model can be trained in an end-to-end manner without having to separately pretrain the attention network or the generative network. The alternate update scheme of the attention and the generative network ensures we can train the whole model end to end. Another merit of our approach is that we have generative modelling of the class conditional distributions, which can help us easily extend our approach to transductive/semi-supervised setting like in [6]. Future research could explore better local visual features, sophisticated generative models like VAEs and improved attention mechanism.

