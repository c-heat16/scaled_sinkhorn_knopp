# Scaled Sinkhorn-Knopp
A distribution-aware cluster assignment mechanism.

Scaled Sinkhorn-Knopp yields more aligned cluster assignments compared to previous distribution-aware cluster 
assignment mechanisms such as [SoLar](https://github.com/hbzju/SoLar/), particularly in the face of poor, or 
low-confidence initial cluster predictions. SoLar and our Scaled Sinkhorn-Knopp were used to cluster 500 batches of 
synthetic predictions of varying quality, denoted by µ, with a batch size of 256, k=24 clusters and imbalance ratio of 
γ=4, with the average resulting *assigned* distribution presented in the figure below. Given a prescribed distribution 
β = {β0, β1, ..., βk}, a set of synthetic predictions P of shape (N, k) are created such that Nxβi records have 
probability µ of being assigned to cluster i. As such, larger values of µ denote *stronger* predictions, with µ=-1 
used to denote random predictions. 

![Prescribed vs assigned distribution](figures/prescribed_vs_assigned_distribution.png)

### Example Usage
An example of how the Scaled Sinkhorn-Knopp cluster assignment mechanism can be used is included in 
[example.py](example.py) and also presented below:
```python
from scaled_sinkhorn_knopp import scaled_sinkhorn_knopp, make_informed_probs, \
    make_linear_sinkhorn_beta, make_cluster_vec

gamma = 4
bz, k, mu = 256, 10, 0.2

beta = make_linear_sinkhorn_beta(k, gamma)
q = make_informed_probs(bz, k, mu, beta)

cluster_assignment_probs = scaled_sinkhorn_knopp(q.clone(), 3, beta, cuda=False)
cluster_assignments = make_cluster_vec(cluster_assignment_probs, k)

print('beta: {}'.format(beta))
print('cluster_assignments: {}'.format(cluster_assignments))
```

### Long-tail Partial-label-learning
The [SoLar](https://github.com/hbzju/SoLar/) codebase for LT-PLL can be easily repurposed to utilize the Scaled 
Sinkhorn-Knopp cluster assignment mechanism. Specifically, line 275 in 
[SoLar/train.py](https://github.com/hbzju/SoLar/blob/main/train.py) should be updated as described below, where
`sinkhorn` is the cluster assignment mechanism proposed by SoLar, and `scaled_sinkhorn_knopp` is our proposed
Scaled Sinkhorn-Knopp.

```python
# old
pseudo_label_soft, flag = sinkhorn(prediction_queue, args.lamd, r_in=emp_dist)

# new
pseudo_label_soft = scaled_sinkhorn_knopp(prediction_queue, args.lamd, emp_dist)
```

Experiments can then be performed using the same commands provided in the [SoLar](https://github.com/hbzju/SoLar/) repo, i.e.:
```shell
CUDA_VISIBLE_DEVICES=0 python -u train.py --exp-dir experiment/CIFAR-10 \
  --dataset cifar10 --num-class 10 --partial_rate 0.5 --imb_type exp --imb_ratio 100 \
  --est_epochs 100 --rho_range 0.2,0.6 --gamma 0.1,0.01
```
