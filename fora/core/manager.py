from collections.abc import Callable, Hashable
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock

from torch import Tensor

from ..config import FORAConfig


type Feature = Tensor | tuple[Tensor, ...]


@dataclass
class CacheEntry:
    step: int
    signature: Hashable | None
    feature: Feature | None


class CacheManager:
    """Reuse the latest feature for each module and conditioning branch."""

    def __init__(self, config: FORAConfig):
        self.config = config
        self._entries: dict[tuple[str, str | None], CacheEntry] = {}
        self._request_lock = Lock()

    def reset(self) -> None:
        self._entries.clear()

    @contextmanager
    def _request(self):
        if not self._request_lock.acquire(blocking=False):
            raise ValueError("FORA cache manager is already in use")
        self.reset()
        try:
            yield
        finally:
            self.reset()
            self._request_lock.release()

    def wrap(
        self,
        key: str,
        step: int,
        compute_fn: Callable[[], Feature],
        context: str | None = None,
        *,
        input_signature: Hashable | None = None,
    ) -> Feature:
        """Return the current feature. Consumers must treat cached features as read-only."""
        cache_key = (key, context)
        entry = self._entries.get(cache_key)
        if step < 0 or (entry is not None and step <= entry.step):
            raise ValueError("steps must be nonnegative and increase per key and context")

        interval = self.config.interval
        if (
            entry is not None
            and entry.feature is not None
            and entry.step // interval == step // interval
            and entry.signature == input_signature
        ):
            entry.step = step
            return entry.feature

        feature = compute_fn()
        if interval == 1:
            cached = None
        elif isinstance(feature, Tensor):
            cached = feature.detach()
        else:
            cached = tuple(tensor.detach() for tensor in feature)
        self._entries[cache_key] = CacheEntry(step, input_signature, cached)
        return feature
