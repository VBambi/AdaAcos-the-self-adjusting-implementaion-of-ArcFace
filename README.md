# AdaAcos
### online computing of the ideal margin parameter
AdaAcos is a hyperparameter free version of ArcFace (https://arxiv.org/abs/1801.07698).<br>
This is a feat, that AdaCos (https://arxiv.org/abs/1905.00292) has already claimed to have pulled off.<br>
However, eliminating the margin parameter completely seems to have made it fall short of the results  ArcFace can archive (!if correctly tuned!).<br>
This led me to experiments with adaptively scaling the margin parameter instead of the scale parameter s. 
## The math
AdaAcos uses the angular distance of the closest uncorresponding class (theta_false_min) and the distance to ground truth (theta_y)
to compute an angular margin parameter just big enough for the classification to barely fail/succeed.
Regarding batchwise computation, the mean of the samples is used, which means that exactly half of the samples in a batch will be classified correctly.<br>
This is only applicable, when the network has already reached a classification accuracy of 50%, m must not "award" the nework to reach 50% accuracy -> cap m to be bigger than or equal zero<br><br>
The formular for the batchwise m is as follows: $\ m = max(median(min(θ_{i,j≠𝑦}) − θ_{i,j=y}), 0)\$<br>
It is the heart of the proposed method.<br>

To set the scale parameter s, the formular from fixed AdaCos is used. 
They chose pi/4 for the initial theta_y completely out of the blue, which doesn't seem to be the most ideal value.
This was changed to pi/2.3, which was validated to work well on EMNIST and DigiFace-1M. 
For Datasets with less classes (eg. MNIST, C=10), the formula will not produce a good s-Parameter.
It seems that the approximations, that were made by the Authors of AdaCos, are not valid for such a low amount of classes.
I recommend to manually set s to ~6 when testing with MNIST.

## See also...
When working with  "ArcFace" and "AdaCos" metrics initially, the implemenations from this repository were used:
https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py

Since this was the starting point, the PyTorch-implementation of AdaAcos in this repo still slightly resembles the original code, hence it was referenced here.
