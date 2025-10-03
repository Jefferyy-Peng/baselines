# ===== gradient_correlation.py =====
import torch
import numpy as np

def _flatten(t):
    if t is None:
        return None
    return t.detach().float().reshape(-1).cpu()

class GradTracker:
    """
    Track parameter gradients over steps and compute correlations/similarities.
    Use after loss.backward().
    """
    def __init__(self, module, param_attrs=("TR", "TI", "TE")):
        """
        module: e.g., your contrast module (SyMRIParamLayer or wrapper)
        param_attrs: attribute names of the *learned tensors* whose grads you want.
                     If you reparameterize (e.g., log-params), pass those names instead.
        """
        self.module = module
        self.param_attrs = tuple(param_attrs)
        self.buffers = {name: [] for name in self.param_attrs}

    def update(self):
        """Call right after loss.backward()."""
        for name in self.param_attrs:
            p = getattr(self.module, name)  # nn.Parameter
            g = _flatten(p.grad)
            if g is None:
                # no grad this step (maybe frozen or skipped)
                continue
            self.buffers[name].append(g.numpy())

    def _stack(self, name):
        arrs = self.buffers[name]
        if len(arrs) == 0:
            return None
        return np.stack(arrs, axis=0)  # [steps, dim]

    def report(self, print_matrix=True):
        """
        Returns:
          stats: dict with 'pearson' and 'cosine' matrices (param x param).
        """
        names = self.param_attrs
        k = len(names)
        pearson = np.eye(k, dtype=np.float32)
        cosine  = np.eye(k, dtype=np.float32)

        stacks = {n: self._stack(n) for n in names}

        # Build per-parameter time-series by averaging |grad| across dims each step.
        # (You can switch to signed mean if you prefer.)
        series = {n: (np.mean(np.abs(stacks[n]), axis=1) if stacks[n] is not None else None)
                  for n in names}

        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if i == j:
                    continue
                si, sj = series[ni], series[nj]
                # Use overlapping steps only
                T = min(len(si) if si is not None else 0,
                        len(sj) if sj is not None else 0)
                if T < 3:
                    pearson[i, j] = np.nan
                    cosine[i, j] = np.nan
                    continue

                a = si[:T]
                b = sj[:T]

                # Pearson correlation of stepwise gradient magnitudes
                pearson[i, j] = np.corrcoef(a, b)[0, 1]

                # Cosine similarity of flattened grads (concat over steps)
                gi = stacks[ni][:T].reshape(T, -1)
                gj = stacks[nj][:T].reshape(T, -1)
                # mean cosine across steps
                num = (gi * gj).sum(axis=1)
                di  = np.linalg.norm(gi, axis=1) + 1e-12
                dj  = np.linalg.norm(gj, axis=1) + 1e-12
                cosine[i, j] = np.mean(num / (di * dj))

        stats = {"names": names, "pearson": pearson, "cosine": cosine}
        if print_matrix:
            header = "      " + "  ".join(f"{n:>10s}" for n in names)
            print("\n[GradTracker] Pearson correlation (per-step |grad|):")
            print(header)
            for i, ni in enumerate(names):
                row = " ".join(f"{pearson[i,j]:10.3f}" for j in range(len(names)))
                print(f"{ni:>6s} {row}")
            print("\n[GradTracker] Cosine similarity (grad direction, mean over steps):")
            print(header)
            for i, ni in enumerate(names):
                row = " ".join(f"{cosine[i,j]:10.3f}" for j in range(len(names)))
                print(f"{ni:>6s} {row}")
        return stats
