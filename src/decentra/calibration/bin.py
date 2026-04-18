import numpy as np


class BinCalibrator:
    """Bin-level attribution calibration.

    Optimizes bin scores via a combined loss:
    ``(1-lam)*L_pred + lam*L_attr + gamma*||s - s_old||^2``

    Parameters
    ----------
    lam : float, default=0.5
        Trade-off between prediction loss and attribution loss.
    gamma : float, default=0.5
        Regularization toward original bin scores.

    Examples
    --------
    >>> cal = BinCalibrator(lam=0.5, gamma=0.5)
    >>> cal.fit(surr_contribs_tr, bb_shap_tr, y_logit_tr, surr_pred_tr, n_features)
    >>> cal_contribs, cal_pred = cal.transform(surr_contribs_te, surr_pred_te)
    """

    def __init__(self, lam=0.5, gamma=0.5):
        self.lam = lam
        self.gamma = gamma

    def _build_bin_matrix(self, surr_contribs, n_features):
        """Build bin indicator matrix B and feature-bin mapping."""
        all_bin_cols = []
        all_bin_scores_orig = []
        feature_bin_ranges = {}
        # Map: (feature_j, unique_val) -> bin column index
        bin_val_map = {}
        col_idx = 0

        for j in range(n_features):
            vals = np.round(surr_contribs[:, j], 8)
            unique_vals = np.unique(vals)
            if len(unique_vals) <= 1:
                continue
            start = col_idx
            for uv in unique_vals:
                mask = (np.abs(vals - uv) < 1e-7).astype(float)
                all_bin_cols.append(mask)
                all_bin_scores_orig.append(uv)
                bin_val_map[(j, round(float(uv), 8))] = col_idx
                col_idx += 1
            feature_bin_ranges[j] = (start, col_idx)

        if len(all_bin_cols) == 0:
            return None, None, {}, {}

        B = np.column_stack(all_bin_cols)
        s_old = np.array(all_bin_scores_orig)
        return B, s_old, feature_bin_ranges, bin_val_map

    def fit(self, surr_contribs, bb_shap, y_logit, surr_pred, n_features):
        """Fit calibration on training data.

        Parameters
        ----------
        surr_contribs : ndarray of shape (n_train, n_features)
            Surrogate's centered contributions on train data.
        bb_shap : ndarray of shape (n_train, n_features)
            Base model's TreeSHAP on train data.
        y_logit : ndarray of shape (n_train,)
            Teacher's log-odds on train data.
        surr_pred : ndarray of shape (n_train,)
            Surrogate's predictions on train data.
        n_features : int
        """
        self.n_features_ = n_features

        B, s_old, feature_bin_ranges, bin_val_map = self._build_bin_matrix(
            surr_contribs, n_features)

        if B is None:
            self.s_new_ = None
            self.feature_bin_ranges_ = {}
            self.bin_val_map_ = {}
            self.base_value_ = float(surr_pred.mean())
            return self

        K = len(s_old)
        base_value = surr_pred.mean() - (B @ s_old).mean()
        y_centered = y_logit - base_value

        BtB = B.T @ B
        Bty = B.T @ y_centered

        # Attribution loss
        AtA = np.zeros((K, K))
        Aty = np.zeros(K)
        for j, (start, end) in feature_bin_ranges.items():
            B_j = B[:, start:end]
            C_j = B_j - B_j.mean(axis=0, keepdims=True)
            phi_j = bb_shap[:, j]
            CjCj = C_j.T @ C_j
            Cjphi = C_j.T @ phi_j
            for ii, ci in enumerate(range(start, end)):
                for jj, cj in enumerate(range(start, end)):
                    AtA[ci, cj] += CjCj[ii, jj]
                Aty[ci] += Cjphi[ii]

        lhs = (1 - self.lam) * BtB + self.lam * AtA + self.gamma * np.eye(K)
        rhs = (1 - self.lam) * Bty + self.lam * Aty + self.gamma * s_old
        self.s_new_ = np.linalg.solve(lhs, rhs)
        self.s_old_ = s_old
        self.base_value_ = float(base_value)
        self.feature_bin_ranges_ = feature_bin_ranges
        self.bin_val_map_ = bin_val_map

        return self

    def transform(self, surr_contribs, surr_pred):
        """Apply fitted calibration to new data.

        Parameters
        ----------
        surr_contribs : ndarray of shape (n_samples, n_features)
            Surrogate's centered contributions on new data.
        surr_pred : ndarray of shape (n_samples,)
            Surrogate's predictions on new data.

        Returns
        -------
        new_contribs : ndarray of shape (n_samples, n_features)
        new_pred : ndarray of shape (n_samples,)
        """
        if self.s_new_ is None:
            return surr_contribs.copy(), surr_pred.copy()

        n_samples = surr_contribs.shape[0]
        K = len(self.s_new_)

        # Build B matrix for new data using fitted bin mapping
        B = np.zeros((n_samples, K))
        for j, (start, end) in self.feature_bin_ranges_.items():
            vals = np.round(surr_contribs[:, j], 8)
            for i in range(n_samples):
                key = (j, round(float(vals[i]), 8))
                if key in self.bin_val_map_:
                    B[i, self.bin_val_map_[key]] = 1.0
                else:
                    # Unseen bin value: find closest known bin
                    known_vals = [k[1] for k in self.bin_val_map_ if k[0] == j]
                    if known_vals:
                        closest = min(known_vals, key=lambda v: abs(v - vals[i]))
                        B[i, self.bin_val_map_[(j, closest)]] = 1.0

        new_contribs = np.zeros_like(surr_contribs)
        for j, (start, end) in self.feature_bin_ranges_.items():
            contrib = B[:, start:end] @ self.s_new_[start:end]
            new_contribs[:, j] = contrib - contrib.mean()

        new_pred = self.base_value_ + B @ self.s_new_
        return new_contribs, new_pred

    def fit_transform(self, surr_contribs, bb_shap, y_logit, surr_pred,
                      n_features):
        """Fit and transform on the same data (convenience method).

        Note: Using this on test data constitutes data leakage.
        Prefer fit() on train, then transform() on test.
        """
        self.fit(surr_contribs, bb_shap, y_logit, surr_pred, n_features)
        return self.transform(surr_contribs, surr_pred)
