"""Identity-only weak registries for opaque PhasePair capability tokens."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from weakref import ReferenceType, ref


def make_identity_weak_registry(
    lock: RLock,
) -> tuple[
    Callable[[object, object], None],
    Callable[[object], object | None],
    Callable[[], tuple[tuple[object, object], ...]],
    Callable[[], tuple[object, ...]],
    Callable[[object], object | None],
]:
    """Return set/get/items/values operations keyed only by object identity.

    The backing dictionary is keyed by ``id(token)`` and every lookup also
    proves that the stored weak reference resolves to the exact caller object.
    The weakref callback removes an entry only when the callback reference is
    still the one stored at that integer id, which closes id-reuse/ABA cleanup.
    No operation hashes or compares the capability token itself.
    """

    entries: dict[int, tuple[ReferenceType[object], object]] = {}
    object_id = id
    weak_reference = ref
    dict_get = dict.get
    dict_pop = dict.pop
    runtime_error_type = RuntimeError
    tuple_factory = tuple

    def set_value(key: object, value: object) -> None:
        key_id = object_id(key)

        def remove(dead_reference: ReferenceType[object]) -> None:
            with lock:
                current = dict_get(entries, key_id)
                if current is not None and current[0] is dead_reference:
                    dict_pop(entries, key_id, None)

        reference = weak_reference(key, remove)
        with lock:
            current = dict_get(entries, key_id)
            if current is not None:
                current_key = current[0]()
                if current_key is not None and current_key is not key:
                    raise runtime_error_type("live identity registry id collision")
            entries[key_id] = (reference, value)

    def get_value(key: object) -> object | None:
        with lock:
            current = dict_get(entries, object_id(key))
            if current is None or current[0]() is not key:
                return None
            return current[1]

    def items() -> tuple[tuple[object, object], ...]:
        with lock:
            output: list[tuple[object, object]] = []
            for reference, value in tuple_factory(entries.values()):
                key = reference()
                if key is not None:
                    output.append((key, value))
            return tuple_factory(output)

    def values() -> tuple[object, ...]:
        return tuple_factory(value for _, value in items())

    def pop_value(key: object) -> object | None:
        key_id = object_id(key)
        with lock:
            current = dict_get(entries, key_id)
            if current is None or current[0]() is not key:
                return None
            dict_pop(entries, key_id, None)
            return current[1]

    return set_value, get_value, items, values, pop_value


__all__: list[str] = []
