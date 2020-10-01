---
layout: page
title: Privacy-Utility via GAN
description: Generating shareable representation of data via adversarial nets
img: /assets/img/privacy-gan-overview.png
importance: 2
---

The urge for offering personalized products and services, and the need for acquiring new users fuel the practice of data sharing among companies. At the same time, data sharing faces the headwind of new laws emphasizing users’ privacy in data. Under the premise that sharing of data occurs from a giver to a recipient, we propose an approach to generation of representational-data for sharing, which achieves value-addition for the recipient’s tasks, while preserving privacy of users.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/privacy-gan-overview.png' | relative_url }}" alt="" title="example image"/>
    </div>
</div>
<div class="caption">
    <b>How to generate shareable representation of data?</b>: The goal is to develop a black box which takes in raw data, privacy policy and information about downstream task and outputs an appropriate representation of data that achieves desired utility while maintaining privacy.
</div>

### The Black Box

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/pgan-arch.png' | relative_url }}" alt="" title="example image"/>
    </div>
</div>
<div class="caption">
    <b>Privacy and Utility aware Generative adversarial nets</b>: We propose an adversarial framework based on Wasserstien GANs with G acting as a shareable-data generator with privacy and utility regularization to achieve a reasonable trade-off and D acting as an adversarial supervisor for G.
</div>
