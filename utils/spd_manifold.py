from typing import Optional, Tuple, Union
import enum
import warnings
import torch
from geoopt.manifolds.base import Manifold
from geoopt import linalg

__all__ = ["SPD"]


EPS = {torch.float32: 1e-4, torch.float64: 1e-7}

class SPDMetricType(enum.Enum):
    AIM = "AIM"


class SPD(Manifold):
    """
    Manifold of SPD matrices with affine-invariant metric.
    This implementation is inspired by https://github.com/geoopt/geoopt.
    """

    __scaling__ = Manifold.__scaling__.copy()
    name = "SPD"
    ndim = 2
    reversible = False

    def __init__(self, default_metric: Union[str, SPDMetricType] = "AIM"):
        super().__init__()
        self.default_metric = SPDMetricType(default_metric)

    def _affine_invariant_metric(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:

        c = torch.linalg.cholesky(x)
        c_inv = torch.linalg.inv(c)
        e, V = torch.linalg.eigh(c_inv @ y @ c_inv.transpose(-1, -2), 'L')
        eigs = torch.log(e) ** 2
        return torch.sqrt(torch.sum(eigs, dim=-1) + 1e-15)

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = torch.allclose(x, x.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return False, "`x != x.transpose` with atol={}, rtol={}".format(atol, rtol)
        e, _ = torch.linalg.eigh(x, "U")
        ok = (e > -atol).min()
        if not ok:
            return False, "eigenvalues of x are not all greater than 0."
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = torch.allclose(u, u.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u != u.transpose` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        symx = linalg.sym(x)
        return linalg.sym_funcm(symx, torch.abs)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return linalg.sym(u)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x @ self.proju(x, u) @ x.transpose(-1, -2)

    _dist_metric = {
        SPDMetricType.AIM: _affine_invariant_metric,
    }

    def dist(self, x: torch.Tensor, y: torch.Tensor, keepdim=False,) -> torch.Tensor:
        return self._dist_metric[self.default_metric](self, x, y)

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: Optional[torch.Tensor] = None, keepdim=False) -> torch.Tensor:
        if v is None:
            v = u
        inv_x = linalg.sym_invm(x)
        ret = linalg.trace(inv_x @ u @ inv_x @ v)
        if keepdim:
            return torch.unsqueeze(torch.unsqueeze(ret, -1), -1)
        return ret

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inv_x = linalg.sym_invm(x)
        return linalg.sym(x + u + 0.5 * u @ inv_x @ u)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        p_inv_tv = torch.linalg.solve(x, u)
        return x @ torch.matrix_exp(p_inv_tv)

    def logmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        c = torch.linalg.cholesky(x, upper=False)
        c_inv = torch.inverse(c)
        mat = c_inv @ u @ c_inv.transpose(-1, -2)
        return c @ self.logm(mat) @ c.transpose(-1, -2)

    def extra_repr(self) -> str:
        return "default_metric={}".format(self.default_metric)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        X_inv = torch.inverse(x)
        A = linalg.sym_inv_sqrtm2(y @ X_inv)[1]
        return A @ v @ A.transpose(-1, -2)

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        tens = 0.5 * torch.randn(*size, dtype=dtype, device=device)
        tens = linalg.sym(tens)
        tens = linalg.sym_funcm(tens, torch.exp)
        return tens

    def origin(
        self,
        *size: Union[int, Tuple[int]],
        dtype=None,
        device=None,
    ) -> torch.Tensor:
        return torch.diag_embed(torch.ones(*size[:-1], dtype=dtype, device=device))

    def logm(self, X):
        e, V = torch.linalg.eigh(X, 'L')
        U = V @ torch.diag_embed(torch.log(e)) @ V.transpose(-1, -2)
        return U

    def sym(self, x: torch.Tensor):
        return 0.5 * (x.transpose(-1, -2) + x)

    def sym_sqrt_inv(self, x: torch.Tensor):
        e, v = torch.linalg.eigh(x, "L")
        sqrt_e = torch.sqrt(e)
        inv_sqrt_e = torch.reciprocal(sqrt_e)
        return (
            v @ torch.diag_embed(inv_sqrt_e) @ v.transpose(-1, -2),
            v @ torch.diag_embed(sqrt_e) @ v.transpose(-1, -2),
        )

