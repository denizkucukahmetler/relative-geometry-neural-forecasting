

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import tensor
from copy import deepcopy
from scipy.integrate import solve_ivp

class TensorNormalizer:
    ''' Expects a 2d tensor to fit and transform '''
    def __init__(self, standardize=True):
        self.standardize = standardize
        # For z-score standardizing
        self.center = None
        self.std = None
        # For 0 to 1 normalizing
        self.mi = None
        self.range = None

    def _check(self, X, been_fit=False):
        assert len(X.shape) == 2
        if been_fit:
            if self.standardize: assert self.center is not None and self.std is not None
            else: assert self.range is not None and self.mi is not None
            
    def to(self, device):
        for attr in ("center", "std", "mi", "range"):
            t = getattr(self, attr, None)
            if torch.is_tensor(t):
                setattr(self, attr, t.to(device))
        return self

    def fit(self, X):
        self._check(X)
        if self.standardize:
            self.center = X.mean(axis=0)
            self.std = X.std(axis=0)
        else:
            self.mi = X.min(axis=0)[0]
            self.range = X.max(axis=0)[0] - self.mi
        return self

    def transform(self, X):
        self._check(X, been_fit=True)
        return (X - self.center) / self.std if self.standardize else (X - self.mi) / self.range

    def fit_transform(self, X):
        self.fit(X)
        return self, self.transform(X)
    
    def inverse_transform(self, X_scaled: torch.Tensor) -> torch.Tensor:
        self._check(X_scaled, been_fit=True)
        self.to(X_scaled.device)         # <- critical line
        if self.standardize:
            return X_scaled * self.std + self.center
        else:
            return X_scaled * self.range + self.mi

    def inverse_transform_new(self, X_scaled):
        if self.standardize:
            # Ensure proper broadcasting of std and center for multi-dimensional data
            return (X_scaled * self.std.view(1, -1).expand_as(X_scaled)) + \
                   self.center.view(1, -1).expand_as(X_scaled)
        else:
            # Ensure proper broadcasting of range and mi for multi-dimensional data
            return (X_scaled * self.range.view(1, -1).expand_as(X_scaled)) + \
                   self.mi.view(1, -1).expand_as(X_scaled)

    def set_keep_columns(self, indices):
        if self.standardize:
            self.center = self.center[indices]
            self.std = self.std[indices]
        else:
            self.mi = self.mi[indices]
            self.range = self.range[indices]

    @staticmethod
    def debug():
        # Standardize
        original = tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
        scaler, scaled = TensorNormalizer(standardize=True).fit_transform(original)
        unscaled = scaler.inverse_transform(scaled)
        assert torch.equal(original, unscaled)
        # Normalize
        original = tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
        scaler, scaled = TensorNormalizer(standardize=False).fit_transform(original)
        unscaled = scaler.inverse_transform(scaled)
        assert torch.equal(original, unscaled)
TensorNormalizer.debug()