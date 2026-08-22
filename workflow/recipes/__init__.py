"""Named recipes: same runtime, different agents and prompts."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from workflow.recipes.apartments import RECIPE as APARTMENTS
from workflow.recipes.research import RECIPE as RESEARCH
from workflow.recipes.types import Recipe, SpecialistSpec

RECIPES: dict[str, Recipe] = {
    RESEARCH.name: RESEARCH,
    APARTMENTS.name: APARTMENTS,
}

_ALIASES: dict[str, str] = {
    "research": "research",
    "general": "research",
    "default": "research",
    "apartments": "apartments",
    "apartment": "apartments",
    "housing": "apartments",
}

_active: ContextVar[Recipe | None] = ContextVar("active_recipe", default=None)


def list_recipes() -> tuple[Recipe, ...]:
    """All registered recipes (used by --list-workflows)."""
    return tuple(RECIPES.values())


def get_recipe(name: str) -> Recipe:
    """Resolve a recipe by name or alias. Raises ValueError if unknown."""
    key = (name or "").strip().lower()
    canonical = _ALIASES.get(key, key)
    recipe = RECIPES.get(canonical)
    if recipe is None:
        known = ", ".join(sorted(RECIPES))
        raise ValueError(f"unknown workflow {name!r}. Available: {known}")
    return recipe


def active_recipe() -> Recipe:
    """Recipe for the current run, or general research if none was selected."""
    current = _active.get()
    if current is not None:
        return current
    return RESEARCH


@contextmanager
def use_recipe(recipe: Recipe) -> Iterator[Recipe]:
    """Make ``recipe`` the active recipe for this context (planner tools, evaluator prompts)."""
    token = _active.set(recipe)
    try:
        yield recipe
    finally:
        _active.reset(token)


__all__ = [
    "APARTMENTS",
    "RECIPES",
    "RESEARCH",
    "Recipe",
    "SpecialistSpec",
    "active_recipe",
    "get_recipe",
    "list_recipes",
    "use_recipe",
]
