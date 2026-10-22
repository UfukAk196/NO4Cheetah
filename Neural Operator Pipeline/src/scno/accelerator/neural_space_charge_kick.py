import cheetah
import torch
from neuralop.models.base_model import BaseModel

from scno.utils.normalization import (
    input_channels,
    nondimensionalize,
    potential_from_hat,
)


class NeuralSpaceChargeKick(SpaceChargeKick):
    """
    Applies the effect of space charge like `SpaceChargeKick`, but solves the Poisson
    equation on the grid with a trained neural operator instead of the integrated
    Green function convolution.

    Only the grid-level solve (`_solve_poisson_on_grid`) changes owner: charge
    deposition, the field from the potential and the interpolation of the forces to
    the particles stay Cheetah's code. Everything consists of torch operations, so the
    gradient survives from the beam parameters to the end of the lattice -- which is
    what this element is for.

    The operator maps the nondimensionalized density to the nondimensionalized
    potential (see `scno.utils.normalization`): input `(B, 4, nx, ny, nz)` with the
    channels rho_hat and log(E_i/L), output `(B, 1, nx, ny, nz)` with phi_hat. It must
    therefore have been trained with `nondim=True` and without output normalizer.

    :param operator: Trained neural operator, e.g. `FNO.from_checkpoint(folder, name)`.
    :param effect_length: Length over which the effect is applied in meters.
    :param grid_shape: Number of grid points in (x, y, tau) directions. May differ from
        the training resolution -- that is the point of neural operators.
    :param grid_extent_x: Dimensions of the grid on which to compute space-charge, as
        multiples of sigma of the beam in the x direction (dimensionless).
    :param grid_extent_y: Dimensions of the grid on which to compute space-charge, as
        multiples of sigma of the beam in the y direction (dimensionless).
    :param grid_extent_tau: Dimensions of the grid on which to compute space-charge, as
        multiples of sigma of the beam in the tau direction (dimensionless).
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment. If `None` (default), a warning is raised for
        invalid names. Set to `True` to sanitise, or `False` to silence the warning.
    :param metadata: Dictionary of arbitrary, serialisable annotations attached to the
        element. This information is *not* used in simulation.
    :param device: Device on which to create the element's tensors.
    :param dtype: Data type of the element's tensors.
    """

    def __init__(
        self,
        operator: BaseModel,
        effect_length: torch.Tensor,
        grid_shape: tuple[int, int, int] = (32, 32, 32),
        grid_extent_x: torch.Tensor | None = None,
        grid_extent_y: torch.Tensor | None = None,
        grid_extent_tau: torch.Tensor | None = None,
        name: str | None = None,
        sanitize_name: bool | None = None,
        metadata: dict | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            effect_length=effect_length,
            grid_shape=grid_shape,
            grid_extent_x=grid_extent_x,
            grid_extent_y=grid_extent_y,
            grid_extent_tau=grid_extent_tau,
            name=name,
            sanitize_name=sanitize_name,
            metadata=metadata,
            device=device,
            dtype=dtype,
        )
        self.operator = operator

    def _solve_poisson_on_grid(
        self, charge_density: torch.Tensor, cell_size: torch.Tensor
    ) -> torch.Tensor:
        """
        Predicts the potential on the grid with the neural operator.

        :param charge_density: Charge density on the grid in C/m^3 of shape
            `(num_vector, nx, ny, nz)`.
        :param cell_size: Cell size of the grid in meters of shape `(num_vector, 3)`,
            with the longitudinal component already scaled by the Lorentz factor.
        :return: Potential on the grid in V of shape `(num_vector, nx, ny, nz)`.
        """
        rho_hat, scaling = nondimensionalize(charge_density, cell_size)
        # Feed the operator in the dtype of its weights; an operator without weights
        # (e.g. `ExactOperator`) gets the full precision of the charge density.
        weight = next(self.operator.parameters(), None)
        dtype = weight.dtype if weight is not None else charge_density.dtype
        x = input_channels(rho_hat.to(dtype), scaling.normalized_extent)
        phi_hat = self.operator(x)[:, 0].to(charge_density.dtype)
        return potential_from_hat(phi_hat, scaling)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["operator"]
