# Auto Segmentation Criterion (ASG) for pytorch

This repo contains a pytorch implementation of the auto segmentation criterion (ASG), introduced in the paper 
[_Wav2Letter: an End-to-End ConvNet-based Speech Recognition System_](https://arxiv.org/abs/1609.03193) by Facebook.

As mentioned in [this blog post](http://danielgalvez.me/jekyll/update/2018/01/12/wav2letter.html) by Daniel Galvez,
ASG, being an alternative to the connectionist temporal classification (CTC) criterion widely used in deep learning, 
has the advantage of being a globally normalized model without the conditional independence assumption of CTC and the 
potential of playing better with
[WFST](https://en.wikipedia.org/wiki/Finite-state_transducer#Weighted_automata) frameworks. 

Unfortunately, Facebook's implementation in its official 
[wav2letter++](https://github.com/facebookresearch/wav2letter) project is based on the ArrayFire C++ framework, which 
makes experimentation rather difficult. Hence we have ported the ASG implementation in wav2letter++ to pytorch as
C++ extensions.

Our implementation should produce the same result as Facebook's, but the implementation is **completely different**.
For example, in their implementation after doing an alpha recursion during the forward pass, they just brute force the
back-propagation during the backward pass, whereas we do a proper alpha-beta recursion during the forward pass, and
during the backward pass there is no recursion at all. Our implementation has the benefit of much higher parallelism 
potential. Another difference is that we try to use pytorch's native
functions as much as possible, whereas Facebook's implementation is basically a gigantic hand-written C code working
on raw arrays.

In the [doc](doc) folder, you can find the [maths derivation](doc/tech_report.pdf) of our implementation.

## Project status

* [x] CPU (openmp) implementation
* [x] GPU (cuda) implementation
* [x] testing
* [ ] performance tuning and comparison
* [ ] Viterbi decoders 
* [ ] generalization to better integrate with general WFSTs decoders

## Using the project

Ensure pytorch > 1.01 is installed, clone the project and in terminal do

```bash
cd torch_asg
pip install .
```

Tested with python 3.7.1. You need to have suitable C++ toolchain installed. For GPU, you need to have an nVidia card
with compute capability >= 6.

Then in your python code:

```python
import torch
from torch_asg import ASGLoss


def test_run():
    num_labels = 7
    input_batch_len = 6
    num_batches = 2
    target_batch_len = 5
    asg_loss = ASGLoss(num_labels=num_labels,
                       reduction='mean',  # mean (default), sum, none
                       gpu_no_stream_impl=False, # see below for explanation
                       forward_only=False # see below for explanation                      
                       )
    for i in range(1):
        # Note that inputs follows the CTC convention so that the batch dimension is 1 instead of 0,
        # in order to have a more efficient GPU implementation
        inputs = torch.randn(input_batch_len, num_batches, num_labels, requires_grad=True)
        targets = torch.randint(0, num_labels, (num_batches, target_batch_len))
        input_lengths = torch.randint(1, input_batch_len + 1, (num_batches,))
        target_lengths = torch.randint(1, target_batch_len + 1, (num_batches,))
        loss = asg_loss.forward(inputs, targets, input_lengths, target_lengths)
        print('loss', loss)
        # You can get the transition matrix if you need it.
        # transition[i, j] is transition score from label j to label i.
        print('transition matrix', asg_loss.transition)
        loss.backward()
        print('transition matrix grad', asg_loss.transition.grad)
        print('inputs grad', inputs.grad)

test_run()
```

There are two options for the loss constructor that warrants further explanation:

* `gpu_no_stream_impl`: by default, if you are using GPU, we are using an implementation that is highly concurrent by
  doing some rather complicated CUDA streams manipulation. You can turn this concurrent implementation off by setting
  this parameter to true, and then CUDA kernel launches are serial. Useful for debugging.
* `forward_only`: by default, our implementation does quite a lot of work during the forward pass concurrently that is
  only useful for calculating the gradients. If you don't need the gradient, setting this parameter to true will give
  a further speed boost. Note that the forward-only mode is automatically active when your model is in evaluation mode.
  
Compared to Facebook's implementation, we have also omitted scaling based on input/output lengths. If you need it, you
can do it yourself by using the `None` reduction and scale the individual scores before summing/averaging.