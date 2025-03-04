import torch.nn
from typing import Tuple, Union, Optional
from geoopt.utils import size2shape
from geoopt.manifolds.base import Manifold
from geoopt.tensor import ManifoldTensor
import torch


class BiOrthogonal(Manifold):
    """
    This class implements the relevant operations for Riemannian Optimization on the Biorthogonal manifold.
    A point [Phi; Psi] for two nxr-matrices fulfills Psi.T Phi = eye(r).

    This implementation is inspired by https://github.com/geoopt/geoopt.
    """

    ndim = 2
    name = 'biorthogonal'

    def __init__(self, n, r, dev=None, dtype=torch.float32):
        super().__init__()
        self.n = n
        self.r = r
        self.eye_r = torch.eye(r, device=dev, dtype=dtype)

    def _split_representation(self, x):
        Phi, Psi = torch.split(x, [self.n, self.n], dim=-2)
        return Phi, Psi

    def _concat_representation(self, Phi, Psi):
        return torch.cat((Phi, Psi), dim=-2)

    def _check_point_on_manifold(
            self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        Phi, Psi = self._split_representation(x)
        ytx = Psi.transpose(-1, -2) @ Phi
        ytx[..., torch.arange(self.r), torch.arange(self.r)] -= 1
        ok = torch.allclose(ytx, ytx.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`Psi^T Phi != I` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def random(self, *size, dtype=torch.float32, device=None) -> torch.Tensor:

        self._assert_check_shape(size2shape(*size), "x")

        tens = torch.zeros((size2shape(*size)[1], size2shape(*size)[0]), device=device, dtype=dtype)
        torch.nn.init.xavier_normal_(tens, 2**(-0.5)*1.4)
        tens2 = tens.transpose(-1, -2) @ torch.linalg.inv(tens @ tens.transpose(-1, -2))
        p = self._concat_representation(tens2, tens.transpose(-1, -2))

        return ManifoldTensor(p, manifold=self)

    def origin(self, *size, dtype=None, device=None, seed=42) -> torch.Tensor:
        self._assert_check_shape(size2shape(*size), "x")
        eye = torch.zeros(*size, dtype=dtype, device=device)
        eye[..., torch.arange(eye.shape[-1]), torch.arange(eye.shape[-1])] += 1
        p = self._concat_representation(eye, eye)
        return ManifoldTensor(p, manifold=self)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Projection of u onto x's tangent space.
        :param x:
        :param u:
        :return:
        """

        Phi, Psi = self._split_representation(x)
        X, Y = self._split_representation(u)

        a_sylv = Psi.transpose(-1, -2) @ Psi
        b_sylv = Phi.transpose(-1, -2) @ Phi
        c_sylv = Y.transpose(-1, -2) @ Phi + Psi.transpose(-1, -2) @ X
        A = self.sylvester_solver(a_sylv, -b_sylv, c_sylv)

        proj = self._concat_representation(X - Psi @ A, Y - Phi @ A.transpose(-1, -2))

        return proj

    def sylvester_solver(self, A, B, C):
        m = B.shape[-1];
        n = A.shape[-1];
        R, U = torch.linalg.eigh(A)
        S, V = torch.linalg.eigh(B)
        F = torch.linalg.solve(U, (C) @ V)
        W = R[..., :, None] - S[..., None, :]
        Y = F / W
        X = U[..., :n, :n] @ Y[..., :n, :m] @ torch.linalg.inv(V)[..., :m, :m]
        return X

    egrad2rgrad = proju

    def inner(
            self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        # inner product
        X_1, Y_1 = self._split_representation(u)
        if v is None:
            X_2 = X_1
            Y_2 = Y_1
        else:
            X_2, Y_2 = self._split_representation(v)
        inner_prod = torch.trace(X_1.transpose(-1, -2) @ X_2) + torch.trace(Y_1.transpose(-1, -2) @ Y_2)
        return inner_prod

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # first order retraction?
        Phi, Psi = self._split_representation(x)
        X, Y = self._split_representation(u)
        R1 = (Phi + X) @ torch.inverse((Psi + Y).transpose(-1, -2) @ (Phi + X))
        R2 = Psi + Y
        return self._concat_representation(R1, R2)

    expmap = retr

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # projection-based parallel-transport
        u = self.proju(y, v)
        return u

    def retr_transp(
            self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # naive way: first retract then transport
        y = self.retr(x, u)
        w = self.transp(x, y, v)
        return y, w

    expmap_transp = retr_transp

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        print('never used')
        return x

    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5) -> Union[Tuple[bool, Optional[str]], bool]:
        print('never used')
        return True, None
