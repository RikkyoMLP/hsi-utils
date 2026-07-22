"""Optional Weights & Biases experiment logging.

The wrapper keeps the rest of an application independent from the ``wandb``
module and becomes a no-op when tracking is disabled or initialization fails.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from importlib import import_module
from numbers import Real
from pathlib import Path
from typing import Any, ParamSpec, Self, TypeVar

from omegaconf import DictConfig, OmegaConf


WandbConfig = Mapping[str, Any] | DictConfig | str | Path | None
P = ParamSpec("P")
R = TypeVar("R")

_ACTIVE_WANDB: ContextVar[tuple[Any, int | None] | None] = ContextVar(
    "hsi_utils_active_wandb", default=None
)


def _as_dict(value: Any, *, label: str) -> dict[str, Any]:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _load_settings(settings: WandbConfig) -> dict[str, Any] | None:
    if settings is None:
        return None
    if isinstance(settings, (str, Path)):
        path = Path(settings)
        if not path.exists():
            return None
        return _as_dict(OmegaConf.load(path), label="W&B settings")
    return _as_dict(settings, label="W&B settings")


def _scalar(value: Any) -> bool | int | float | None:
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Real):
        return float(value)
    if getattr(value, "ndim", None) == 0:
        item = getattr(value, "item", None)
        if callable(item):
            return _scalar(item())
    return None


def _sequence(value: Any) -> list[Any] | None:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    if isinstance(value, Sequence):
        return list(value)
    return None


def _numeric_sequence(value: Any) -> list[bool | int | float] | None:
    values = _sequence(value)
    if values is None:
        return None
    numbers = [_scalar(item) for item in values]
    if any(item is None for item in numbers):
        return None
    return [item for item in numbers if item is not None]


def _metric_items(namespace: str, key: str, value: Any) -> dict[str, Any]:
    scalar = _scalar(value)
    if scalar is not None:
        return {f"{namespace}/{key}": scalar}

    if isinstance(value, Mapping):
        metrics: dict[str, Any] = {}
        for child_key, child_value in value.items():
            metrics.update(
                _metric_items(
                    namespace,
                    f"{key}/{child_key}",
                    child_value,
                )
            )
        return metrics

    values = _numeric_sequence(value)
    if values is None:
        return {}
    if key.endswith("_per_scene"):
        metric = key.removesuffix("_per_scene")
        return {
            f"{namespace}/{metric}_S{index:02d}": item
            for index, item in enumerate(values, start=1)
        }
    if key.startswith("stage_") and key.endswith("_mean"):
        metric = key.removeprefix("stage_").removesuffix("_mean")
        return {
            f"{namespace}/stage_{index}_{metric}": item
            for index, item in enumerate(values, start=1)
        }
    return {
        f"{namespace}/{key}_{index:02d}": item
        for index, item in enumerate(values, start=1)
    }


class WandbLogger:
    """Small, failure-tolerant wrapper around a W&B run.

    Use :meth:`from_config` to create an instance. Disabled and failed
    instances safely ignore calls to :meth:`log` and :meth:`finish`, which
    keeps optional experiment tracking out of the training control flow.
    """

    def __init__(
        self,
        *,
        run: Any | None = None,
        backend: Any | None = None,
        error: Exception | None = None,
    ) -> None:
        self._run = run
        self._backend = backend
        self._error = error
        self._logged_once: set[str] = set()
        run_id = getattr(run, "id", None)
        self._id = str(run_id) if run_id is not None else None

    @classmethod
    def from_config(
        cls,
        settings: WandbConfig,
        *,
        run_name: str | None = None,
        run_config: Mapping[str, Any] | DictConfig | None = None,
        run_id: str | None = None,
        resume: str | bool | None = "must",
        default_project: str | None = None,
        api_key: str | None = None,
        strict: bool = False,
    ) -> Self:
        """Initialize W&B from a YAML path or settings mapping.

        ``enabled``, ``login``, and ``api_key_env`` are wrapper settings. All
        remaining settings are forwarded to :func:`wandb.init`. ``run_name``,
        ``run_config``, and ``run_id`` override their matching settings, while
        ``default_project`` only fills a missing project. When ``run_id`` is
        supplied, the run is resumed with ``resume=\"must\"`` by default.

        Missing settings files and ``enabled: false`` return a disabled logger.
        Initialization errors also return a disabled logger unless ``strict``
        is true; the original exception is then available through
        :attr:`error`.
        """

        try:
            init_kwargs = _load_settings(settings)
            if init_kwargs is None:
                return cls()
            if not bool(init_kwargs.pop("enabled", False)):
                return cls()

            login = bool(init_kwargs.pop("login", True))
            api_key_env = str(init_kwargs.pop("api_key_env", "WANDB_API_KEY"))
            if init_kwargs.get("entity") == "":
                init_kwargs["entity"] = None
            if default_project is not None:
                init_kwargs.setdefault("project", default_project)
            if run_name is not None:
                init_kwargs["name"] = run_name
            if run_config is not None:
                init_kwargs["config"] = _as_dict(
                    run_config, label="W&B run config"
                )
            if run_id:
                init_kwargs["id"] = str(run_id)
                if resume is not None:
                    init_kwargs["resume"] = resume

            backend = import_module("wandb")
            if login:
                resolved_key = api_key or os.environ.get(api_key_env)
                if resolved_key:
                    backend.login(key=resolved_key)
                else:
                    backend.login()

            run = backend.init(**init_kwargs)
            if run is None:
                raise RuntimeError("wandb.init() did not return a run")
            return cls(run=run, backend=backend)
        except Exception as exc:
            if strict:
                raise
            return cls(error=exc)

    @property
    def enabled(self) -> bool:
        """Whether this logger currently has an active W&B run."""

        return self._run is not None

    @property
    def error(self) -> Exception | None:
        """Initialization error, if W&B failed to start."""

        return self._error

    @property
    def id(self) -> str | None:
        """Run ID, retained after :meth:`finish` for checkpoint metadata."""

        return self._id

    @property
    def run(self) -> Any | None:
        """Underlying W&B run for features not covered by this wrapper."""

        return self._run

    def log(
        self,
        data: Mapping[str, Any],
        *,
        step: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Log metrics or media when tracking is active."""

        if self._run is None:
            return
        log_kwargs = dict(kwargs)
        if step is not None:
            log_kwargs["step"] = step
        self._run.log(dict(data), **log_kwargs)

    def log_result(
        self,
        namespace: str,
        result: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        """Log a decorated function result using standard naming conventions.

        Scalars are placed below ``namespace``. Numeric ``*_per_scene`` and
        ``stage_*_mean`` sequences are expanded into individual charts. Image
        sequences named ``*_rgb`` become ``<name>/Sxx`` media panels; ``gt``
        images are uploaded only once per logger instance.
        """

        if self._run is None:
            return
        if not isinstance(result, Mapping):
            raise TypeError("A wandb-captured function must return a mapping")

        normalized_namespace = namespace.strip("/")
        if not normalized_namespace:
            raise ValueError("W&B result namespace cannot be empty")

        log_data: dict[str, Any] = {}
        logged_once: set[str] = set()
        scene_psnr = _numeric_sequence(result.get("psnr_per_scene"))
        for raw_key, value in result.items():
            key = str(raw_key)
            if not key.endswith("_rgb"):
                log_data.update(
                    _metric_items(normalized_namespace, key, value)
                )
                continue

            media_namespace = key.removesuffix("_rgb").strip("/")
            if not media_namespace or media_namespace in self._logged_once:
                continue
            images = _sequence(value)
            if not images:
                continue
            for index, data in enumerate(images, start=1):
                scene = f"S{index:02d}"
                if (
                    media_namespace == "recon"
                    and scene_psnr
                    and index <= len(scene_psnr)
                ):
                    caption = f"{scene} PSNR={scene_psnr[index - 1]:.2f}dB"
                elif media_namespace == "gt":
                    caption = f"{scene} GT"
                else:
                    caption = f"{scene} {media_namespace}"
                log_data[f"{media_namespace}/{scene}"] = self.image(
                    data, caption=caption
                )
            if media_namespace == "gt":
                logged_once.add(media_namespace)

        if not log_data:
            return
        self.log(log_data, step=step)
        self._logged_once.update(logged_once)

    def image(self, data: Any, **kwargs: Any) -> Any:
        """Create a W&B image, or return ``data`` when tracking is inactive."""

        if self._run is None or self._backend is None:
            return data
        return self._backend.Image(data, **kwargs)

    def finish(self, **kwargs: Any) -> None:
        """Finish the active run. Repeated calls are safe."""

        if self._run is None:
            return
        run = self._run
        run.finish(**kwargs)
        self._run = None

    @contextmanager
    def autolog(self, *, step: int | None = None) -> Iterator[Self]:
        """Activate this logger for :func:`wandb_capture` calls in the block."""

        token = _ACTIVE_WANDB.set((self, step))
        try:
            yield self
        finally:
            _ACTIVE_WANDB.reset(token)

    def __bool__(self) -> bool:
        return self.enabled

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.finish()


def wandb_capture(namespace: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Automatically log the mapping returned by a function.

    The decorated function behaves normally outside ``WandbLogger.autolog``.
    Inside an active block, its return mapping is converted and uploaded at the
    block's step without embedding W&B-specific calls in the function body.
    """

    normalized_namespace = namespace.strip("/")
    if not normalized_namespace:
        raise ValueError("W&B capture namespace cannot be empty")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = func(*args, **kwargs)
            active = _ACTIVE_WANDB.get()
            if active is not None:
                logger, step = active
                logger.log_result(normalized_namespace, result, step=step)
            return result

        return wrapper

    return decorator
