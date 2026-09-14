from __future__ import annotations

from collections.abc import Sequence

from jax import Array

# helper functions for the trim routine to create groups of control surfaces or thrust nodes who should share a common
# value


def group_key(group: Sequence[str]) -> str:
    """create key for a group of names (thrust nodes or control surfaces)."""
    return "+".join(group)


def expand_groups(
    group_values: dict[str, Array],
    groups: Sequence[Sequence[str]],
) -> dict[str, Array]:
    """Expand a group-keyed dict into a per-member dict so tied members share one value."""
    per_member: dict[str, Array] = {}
    for group in groups:
        v = group_values[group_key(group)]
        for member in group:
            per_member[member] = v
    return per_member


def parse_groups(
    value: Sequence[str | Sequence[str]] | str | None,
    arg_name: str,
) -> list[tuple[str, ...]]:
    r"""
    Convert a user-specified group of control surfaces or thrust nodes into a list of tuples, where each tuple contains
    the names of the members in that group.
    :param value: Input value for groupings.
    :param arg_name: Name of the argument being parsed.
    :return: List of tuples containing the names of the members in each group.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [(value,)]
    if isinstance(value, Sequence):
        groups: list[tuple[str, ...]] = []
        for item in value:
            if isinstance(item, str):
                groups.append((item,))
            elif isinstance(item, Sequence):
                if not item:
                    raise ValueError(
                        f"Each {arg_name} group must contain at least one key."
                    )
                groups.append(tuple(item))
            else:
                raise ValueError(
                    f"{arg_name} entries must be a string or sequence of strings. Got {type(item)}."
                )
        return groups
    raise ValueError(
        f"{arg_name} must be a string, a sequence, or None. Got {type(value)}."
    )


def check_unique_members(groups: Sequence[Sequence[str]], arg_name: str) -> None:
    r"""
    Ensure that each entry only appears in one group. Raises a ValueError if any entry appears in more than one group.
    :param groups: List of groups to check.
    :param arg_name: Name of the argument being parsed.
    """
    seen: set[str] = set()
    for group in groups:
        for member in group:
            if member in seen:
                raise ValueError(
                    f"{arg_name} entry {member!r} appears in more than one trim group."
                )
            seen.add(member)
