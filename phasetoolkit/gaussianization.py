"""Phase-wise gaussianization transforms for action spaces."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, Literal, Optional, Tuple, Any, Union
from pathlib import Path

import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import FastICA
import pickle


ArrayDict = Dict[int, np.ndarray]


def _safe_cov(arr: np.ndarray) -> np.ndarray:
    if arr.shape[0] < 2:
        return np.eye(arr.shape[1])
    return np.cov(arr, rowvar=False)


class PhaseGaussianizerError(RuntimeError):
    """Raised when gaussianization cannot be performed."""


@dataclass
class PhaseTransformResult:
    transformed: np.ndarray
    log_det_jacobian: Optional[np.ndarray] = None
    diagnostics: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class GaussianizerSpec:
    """Configuration wrapper for building gaussianizers from presets."""

    label: str
    kind: Literal["ica", "realnvp"]
    config: Optional[Any] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


class BasePhaseGaussianizer:
    """Common interface for phase-wise gaussianization."""

    name: str = "base"

    def __init__(self, action_dim: Optional[int] = None) -> None:
        self._action_dim = action_dim
        self._phase_ids: set[int] = set()

    @property
    def action_dim(self) -> Optional[int]:
        return self._action_dim

    def available_phases(self) -> Tuple[int, ...]:
        return tuple(sorted(self._phase_ids))

    def fit(self, phase_actions: ArrayDict) -> None:
        if not phase_actions:
            raise PhaseGaussianizerError("No phase actions supplied for fitting")
        for phase_id, actions in phase_actions.items():
            self._validate_actions(actions, expect_dim=self._action_dim)
            self._phase_ids.add(int(phase_id))
            self._fit_single_phase(int(phase_id), actions)
        if self._action_dim is None:
            first = next(iter(phase_actions.values()))
            self._action_dim = first.shape[1]

    def transform(self, phase_id: int, actions: np.ndarray) -> PhaseTransformResult:
        self._ensure_phase_known(phase_id)
        self._validate_actions(actions, expect_dim=self._action_dim)
        return self._transform_single_phase(int(phase_id), actions)

    def inverse_transform(self, phase_id: int, canonical_actions: np.ndarray) -> np.ndarray:
        self._ensure_phase_known(phase_id)
        self._validate_actions(canonical_actions, expect_dim=self._action_dim)
        return self._inverse_single_phase(int(phase_id), canonical_actions)

    def diagnostics(self, phase_id: int) -> Dict[str, Any]:
        self._ensure_phase_known(phase_id)
        return self._diagnostics_single_phase(int(phase_id))

    def state_dict(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _fit_single_phase(self, phase_id: int, actions: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _transform_single_phase(self, phase_id: int, actions: np.ndarray) -> PhaseTransformResult:  # pragma: no cover - interface
        raise NotImplementedError

    def _inverse_single_phase(self, phase_id: int, canonical_actions: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def _diagnostics_single_phase(self, phase_id: int) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def _validate_actions(actions: np.ndarray, expect_dim: Optional[int] = None) -> None:
        if actions.ndim != 2:
            raise PhaseGaussianizerError("Actions must be a 2D array")
        if expect_dim is not None and actions.shape[1] != expect_dim:
            raise PhaseGaussianizerError(
                f"Expected action dimension {expect_dim}, got {actions.shape[1]}"
            )

    def _ensure_phase_known(self, phase_id: int) -> None:
        if int(phase_id) not in self._phase_ids:
            raise PhaseGaussianizerError(f"Phase {phase_id} has not been fitted")


@dataclass
class ICAGaussianizerConfig:
    whiten: Any = "unit-variance"
    max_iter: int = 1000
    tolerance: float = 1e-4
    random_state: Optional[int] = None


@dataclass
class _ICAPhaseState:
    ica: FastICA
    mean: np.ndarray
    whitening_matrix: Optional[np.ndarray]
    dewhitening_matrix: Optional[np.ndarray]
    canonical_std: Optional[np.ndarray] = None


class ICAPhaseGaussianizer(BasePhaseGaussianizer):
    """Linear gaussianization via FastICA with optional whitening."""

    name = "ica"

    def __init__(self, config: Optional[ICAGaussianizerConfig] = None, action_dim: Optional[int] = None) -> None:
        super().__init__(action_dim)
        self.config = config or ICAGaussianizerConfig()
        self._states: Dict[int, _ICAPhaseState] = {}

    def _fit_single_phase(self, phase_id: int, actions: np.ndarray) -> None:
        cfg = self.config
        whiten_param = cfg.whiten
        if whiten_param is True:
            whiten_param = "unit-variance"
        elif whiten_param not in (False, "unit-variance", "arbitrary-variance"):
            raise PhaseGaussianizerError(
                f"Unsupported FastICA whitening option: {whiten_param}. Use True, False, 'unit-variance', or 'arbitrary-variance'."
            )

        ica = FastICA(
            whiten=whiten_param,
            max_iter=cfg.max_iter,
            tol=cfg.tolerance,
            random_state=cfg.random_state,
        )
        transformed = ica.fit_transform(actions)
        mean = ica.mean_.copy()
        whitening_matrix = getattr(ica, "whitening_", None)
        if whitening_matrix is not None:
            whitening_matrix = whitening_matrix.copy()
        dewhitening = None
        if whitening_matrix is not None:
            dewhitening = np.linalg.pinv(whitening_matrix)
        self._states[phase_id] = _ICAPhaseState(
            ica=ica,
            mean=mean,
            whitening_matrix=whitening_matrix,
            dewhitening_matrix=dewhitening,
        )
        if transformed.shape[0] >= 2:
            diag_cov = np.cov(transformed, rowvar=False)
            diag_std = np.sqrt(np.clip(np.diag(diag_cov), 1e-9, None))
        else:
            diag_cov = np.eye(transformed.shape[1])
            diag_std = np.ones(transformed.shape[1])
        self._states[phase_id].canonical_std = diag_std

    def _transform_single_phase(self, phase_id: int, actions: np.ndarray) -> PhaseTransformResult:
        state = self._states[phase_id]
        canonical = state.ica.transform(actions)
        std = getattr(state, "canonical_std", None)
        log_det = None
        diagnostics = {
            "covariance": _safe_cov(canonical),
            "component_kurtosis": self._component_kurtosis(canonical),
        }
        if std is not None:
            nonzero = np.where(std > 0)[0]
            canonical[:, nonzero] = canonical[:, nonzero] / std[nonzero]
            log_det = -np.sum(np.log(std[nonzero])) * np.ones(canonical.shape[0])
            diagnostics["scaling_std"] = std
        return PhaseTransformResult(transformed=canonical, log_det_jacobian=log_det, diagnostics=diagnostics)

    def _inverse_single_phase(self, phase_id: int, canonical_actions: np.ndarray) -> np.ndarray:
        state = self._states[phase_id]
        std = getattr(state, "canonical_std", None)
        recovered = canonical_actions.copy()
        if std is not None:
            nonzero = np.where(std > 0)[0]
            recovered[:, nonzero] = recovered[:, nonzero] * std[nonzero]
        return state.ica.inverse_transform(recovered)

    def _diagnostics_single_phase(self, phase_id: int) -> Dict[str, Any]:
        state = self._states[phase_id]
        diag = {
            "mean": state.mean,
            "mixing_matrix": state.ica.mixing_,
            "components": state.ica.components_,
            "whitening": state.whitening_matrix,
        }
        if hasattr(state, "canonical_std"):
            diag["canonical_std"] = getattr(state, "canonical_std")
        return diag

    def state_dict(self) -> Dict[str, Any]:
        payload = {
            "action_dim": self._action_dim,
            "config": asdict(self.config),
            "phases": {},
        }
        for phase_id, state in self._states.items():
            payload["phases"][int(phase_id)] = {
                "ica": pickle.dumps(state.ica),
                "mean": state.mean,
                "whitening_matrix": state.whitening_matrix,
                "dewhitening_matrix": state.dewhitening_matrix,
                "canonical_std": state.canonical_std,
            }
        return payload

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._states = {}
        self._phase_ids = set()
        self._action_dim = state_dict.get("action_dim")
        cfg = state_dict.get("config", {})
        self.config = ICAGaussianizerConfig(**cfg)
        for phase_id, payload in state_dict.get("phases", {}).items():
            ica = pickle.loads(payload["ica"])
            state = _ICAPhaseState(
                ica=ica,
                mean=payload["mean"],
                whitening_matrix=payload.get("whitening_matrix"),
                dewhitening_matrix=payload.get("dewhitening_matrix"),
                canonical_std=payload.get("canonical_std"),
            )
            self._states[int(phase_id)] = state
            self._phase_ids.add(int(phase_id))

    @staticmethod
    def _component_kurtosis(arr: np.ndarray) -> np.ndarray:
        centered = arr - arr.mean(axis=0, keepdims=True)
        std = np.sqrt(np.clip(centered.var(axis=0), 1e-9, None))
        normed = centered / std
        return np.mean(normed**4, axis=0) - 3.0


@dataclass
class RealNVPConfig:
    num_coupling_layers: int
    hidden_dim: int
    max_epochs: int
    batch_size: int
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip: Optional[float] = 5.0
    scale_activation: Literal["tanh", "identity"] = "tanh"
    patience: Optional[int] = None

    @staticmethod
    def preset(name: Literal["light", "medium", "heavy"]) -> "RealNVPConfig":
        presets = {
            "light": RealNVPConfig(num_coupling_layers=4, hidden_dim=64, max_epochs=150, batch_size=256),
            "medium": RealNVPConfig(num_coupling_layers=8, hidden_dim=128, max_epochs=250, batch_size=512),
            "heavy": RealNVPConfig(num_coupling_layers=12, hidden_dim=256, max_epochs=400, batch_size=512),
        }
        return presets[name]


@dataclass
class _FlowPhaseState:
    flow: "_RealNVPFlow"
    mean: np.ndarray
    std: np.ndarray
    training_log: Dict[str, Any] = field(default_factory=dict)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        return self.net(x)


class _RealNVPCoupling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mask: torch.Tensor,
        scale_activation: Literal["tanh", "identity"] = "tanh",
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        condition_dim = int(mask.sum().item())
        transform_dim = input_dim - condition_dim
        if transform_dim <= 0:
            raise PhaseGaussianizerError("Coupling mask must leave dimensions to transform")
        self.scale_net = _MLP(condition_dim, hidden_dim, transform_dim)
        self.translate_net = _MLP(condition_dim, hidden_dim, transform_dim)
        self.scale_activation = scale_activation
        self.condition_dim = condition_dim
        self.transform_dim = transform_dim

    def forward(self, x: torch.Tensor, inverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask
        condition = x[:, mask.bool()]
        s_vals = self.scale_net(condition)
        t_vals = self.translate_net(condition)
        if self.scale_activation == "tanh":
            s_vals = torch.tanh(s_vals)
        y = x.clone()
        idx = (~mask.bool()).nonzero(as_tuple=False).squeeze(-1)
        if inverse:
            y_partial = (x[:, idx] - t_vals) * torch.exp(-s_vals)
            log_det = -s_vals.sum(dim=1)
        else:
            y_partial = x[:, idx] * torch.exp(s_vals) + t_vals
            log_det = s_vals.sum(dim=1)
        y[:, idx] = y_partial
        return y, log_det


class _RealNVPFlow(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        hidden_dim: int,
        scale_activation: Literal["tanh", "identity"] = "tanh",
    ) -> None:
        super().__init__()
        masks = self._build_masks(input_dim, num_layers)
        layers = []
        for mask in masks:
            layers.append(
                _RealNVPCoupling(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    mask=mask,
                    scale_activation=scale_activation,
                )
            )
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def _build_masks(input_dim: int, num_layers: int) -> Iterable[torch.Tensor]:
        base = torch.tensor([(i % 2) for i in range(input_dim)], dtype=torch.float32)
        masks = []
        for i in range(num_layers):
            if i % 2 == 0:
                mask = base.clone()
            else:
                mask = 1.0 - base
            masks.append(mask)
        return masks

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        out = x
        for layer in self.layers:
            out, log_det = layer(out, inverse=False)
            log_det_total = log_det_total + log_det
        return out, log_det_total

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        out = z
        for layer in reversed(self.layers):
            out, log_det = layer(out, inverse=True)
            log_det_total = log_det_total + log_det
        return out, log_det_total


class RealNVPPhaseGaussianizer(BasePhaseGaussianizer):
    """Non-linear gaussianization via RealNVP normalizing flows."""

    name = "realnvp"

    def __init__(
        self,
        config: RealNVPConfig,
        action_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(action_dim)
        self.config = config
        self.device = device or torch.device("cpu")
        self._states: Dict[int, _FlowPhaseState] = {}

    def _fit_single_phase(self, phase_id: int, actions: np.ndarray) -> None:
        mean = actions.mean(axis=0)
        std = np.std(actions, axis=0)
        std = np.where(std < 1e-6, 1e-6, std)
        normalized = (actions - mean) / std
        flow = _RealNVPFlow(
            input_dim=normalized.shape[1],
            num_layers=self.config.num_coupling_layers,
            hidden_dim=self.config.hidden_dim,
            scale_activation=self.config.scale_activation,
        ).to(self.device)
        training_log = self._train_flow(flow, normalized)
        self._states[phase_id] = _FlowPhaseState(
            flow=flow,
            mean=mean,
            std=std,
            training_log=training_log,
        )

    def _transform_single_phase(self, phase_id: int, actions: np.ndarray) -> PhaseTransformResult:
        state = self._states[phase_id]
        normalized = (actions - state.mean) / state.std
        torch_input = torch.from_numpy(normalized.astype(np.float32)).to(self.device)
        with torch.no_grad():
            z, log_det_flow = state.flow(torch_input)
        log_det_affine = -np.sum(np.log(state.std))
        combined_log_det = log_det_flow.cpu().numpy() + log_det_affine
        diagnostics = {
            "mean": state.mean,
            "std": state.std,
            "training": state.training_log,
            "covariance": _safe_cov(z.cpu().numpy()),
        }
        return PhaseTransformResult(
            transformed=z.cpu().numpy(),
            log_det_jacobian=combined_log_det,
            diagnostics=diagnostics,
        )

    def _inverse_single_phase(self, phase_id: int, canonical_actions: np.ndarray) -> np.ndarray:
        state = self._states[phase_id]
        torch_input = torch.from_numpy(canonical_actions.astype(np.float32)).to(self.device)
        with torch.no_grad():
            x_norm, _ = state.flow.inverse(torch_input)
        actions = x_norm.cpu().numpy() * state.std + state.mean
        return actions

    def _diagnostics_single_phase(self, phase_id: int) -> Dict[str, Any]:
        state = self._states[phase_id]
        diag = dict(state.training_log)
        diag.update({
            "mean": state.mean,
            "std": state.std,
        })
        return diag

    def _train_flow(self, flow: _RealNVPFlow, normalized_actions: np.ndarray) -> Dict[str, Any]:
        device = self.device
        cfg = self.config
        data = torch.from_numpy(normalized_actions.astype(np.float32))
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(flow.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        flow.train()

        def gaussian_log_prob(z: torch.Tensor) -> torch.Tensor:
            log2pi = torch.log(torch.tensor(2.0 * math.pi, device=z.device))
            return -0.5 * (z.pow(2).sum(dim=1) + z.shape[1] * log2pi)

        best_loss = float("inf")
        epochs_no_improve = 0
        losses = []
        for epoch in range(cfg.max_epochs):
            epoch_losses = []
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                z, log_det = flow(batch)
                log_prob = gaussian_log_prob(z) + log_det
                loss = -log_prob.mean()
                loss.backward()
                if cfg.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(flow.parameters(), cfg.gradient_clip)
                optimizer.step()
                epoch_losses.append(loss.item())
            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            losses.append(epoch_loss)
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if cfg.patience is not None and epochs_no_improve >= cfg.patience:
                break
        flow.eval()
        return {
            "losses": losses,
            "best_loss": best_loss,
            "epochs_trained": len(losses),
        }

    def state_dict(self) -> Dict[str, Any]:
        payload = {
            "action_dim": self._action_dim,
            "config": asdict(self.config),
            "phases": {},
        }
        for phase_id, state in self._states.items():
            payload["phases"][int(phase_id)] = {
                "mean": state.mean,
                "std": state.std,
                "training_log": state.training_log,
                "flow_state": state.flow.state_dict(),
            }
        return payload

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._states = {}
        self._phase_ids = set()
        self._action_dim = state_dict.get("action_dim")
        cfg = state_dict.get("config", {})
        self.config = RealNVPConfig(**cfg)
        for phase_id, payload in state_dict.get("phases", {}).items():
            flow = _RealNVPFlow(
                input_dim=self._action_dim,
                num_layers=self.config.num_coupling_layers,
                hidden_dim=self.config.hidden_dim,
                scale_activation=self.config.scale_activation,
            ).to(self.device)
            flow.load_state_dict(payload["flow_state"])
            state = _FlowPhaseState(
                flow=flow,
                mean=payload["mean"],
                std=payload["std"],
                training_log=payload.get("training_log", {}),
            )
            self._states[int(phase_id)] = state
            self._phase_ids.add(int(phase_id))


def build_gaussianizer(
    kind: Literal["ica", "realnvp"],
    **kwargs: Any,
) -> BasePhaseGaussianizer:
    if kind == "ica":
        cfg = kwargs.get("config")
        return ICAPhaseGaussianizer(config=cfg)
    if kind == "realnvp":
        cfg = kwargs.get("config")
        if cfg is None:
            raise PhaseGaussianizerError("RealNVP requires a configuration")
        return RealNVPPhaseGaussianizer(config=cfg, device=kwargs.get("device"))
    raise PhaseGaussianizerError(f"Unknown gaussianizer kind '{kind}'")


def build_gaussianizer_from_spec(spec: GaussianizerSpec) -> Tuple[str, BasePhaseGaussianizer]:
    """Instantiate a gaussianizer from a specification object."""

    kwargs = dict(spec.extra_kwargs)
    if spec.config is not None:
        kwargs.setdefault("config", spec.config)
    gaussianizer = build_gaussianizer(spec.kind, **kwargs)
    return spec.label, gaussianizer


__all__ = [
    "PhaseTransformResult",
    "PhaseGaussianizerError",
    "BasePhaseGaussianizer",
    "ICAGaussianizerConfig",
    "ICAPhaseGaussianizer",
    "RealNVPConfig",
    "RealNVPPhaseGaussianizer",
    "build_gaussianizer",
    "GaussianizerSpec",
    "build_gaussianizer_from_spec",
    "save_gaussianizer",
    "load_gaussianizer",
]


def save_gaussianizer(path: Union[str, Path], spec: GaussianizerSpec, gaussianizer: BasePhaseGaussianizer) -> None:
    """Persist a fitted gaussianizer to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "spec": {
            "label": spec.label,
            "kind": spec.kind,
            "config": spec.config,
            "extra_kwargs": spec.extra_kwargs,
        },
        "state": gaussianizer.state_dict(),
    }
    with path.open("wb") as f:
        pickle.dump(state, f)


def load_gaussianizer(path: Union[str, Path]) -> Tuple[GaussianizerSpec, BasePhaseGaussianizer]:
    """Load a gaussianizer and its spec from disk."""
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    spec_payload = payload["spec"]
    spec = GaussianizerSpec(
        label=spec_payload["label"],
        kind=spec_payload["kind"],
        config=spec_payload.get("config"),
        extra_kwargs=spec_payload.get("extra_kwargs", {}),
    )
    _, gaussianizer = build_gaussianizer_from_spec(spec)
    gaussianizer.load_state_dict(payload["state"])
    return spec, gaussianizer
