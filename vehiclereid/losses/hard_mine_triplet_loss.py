from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import torch
from torch.nn import functional as F
import numpy as np

from functools import reduce


def calc_dist_split(qf, gf, split=0):
    qf = qf.cuda()
    m = qf.shape[0]
    n = gf.shape[0]
    distmat = gf.new(m, n)

    if split == 0:
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).cuda().sum(dim=1, keepdim=True).expand(n, m).t()

    # 用于测试时控制显存
    else:
        start = 0
        while start < n:
            end = start + split if (start + split) < n else n
            num = end - start

            sub_distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, num) + \
                    torch.pow(gf[start:end], 2).cuda().sum(dim=1, keepdim=True).expand(num, m).t()
            # sub_distmat.addmm_(1, -2, qf, gf[start:end].t())
            sub_distmat.addmm_(qf, gf[start:end].cuda().t(), beta=1, alpha=-2)
            distmat[:, start:end] = sub_distmat.cpu()
            start += num

    return distmat


def euclidean_dist(x, y):
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist


def _batch_hard(mat_distance, mat_similarity, indice=False):
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, 0]
	hard_p_indice = positive_indices[:, 0]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, 0]
	hard_n_indice = negative_indices[:, 0]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n

class TripletLoss(nn.Module):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin, normalize_feature=False):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

	def forward(self, emb, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb = F.normalize(emb)
		mat_dist = euclidean_dist(emb, emb)
		# mat_dist = cosine_dist(emb, emb)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
		assert dist_an.size(0)==dist_ap.size(0)
		y = torch.ones_like(dist_ap)
		loss = self.margin_loss(dist_an, dist_ap, y)
		prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
		return loss, prec
