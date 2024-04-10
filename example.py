__author__ = 'Connor Heaton'

from scaled_sinkhorn_knopp import scaled_sinkhorn_knopp, make_informed_probs, make_linear_sinkhorn_beta, \
    make_cluster_vec

gamma = 4
bz, k, mu = 256, 10, 0.2

beta = make_linear_sinkhorn_beta(k, gamma)
q = make_informed_probs(bz, k, mu, beta)

cluster_assignment_probs = scaled_sinkhorn_knopp(q.clone(), 3, beta, cuda=False)
cluster_assignments = make_cluster_vec(cluster_assignment_probs, k)

print('beta: {}'.format(beta))
print('cluster_assignments: {}'.format(cluster_assignments))
