from abc import ABC
from dataclasses import dataclass


class ExerciseStyle(ABC):
    """Marker hierarchy — engines dispatch on `isinstance(option.exercise, ...)`.

    Deliberately not an Enum: Bermudan needs to carry its own data (the
    exercise dates), which an Enum member can't do cleanly, and dataclasses
    let all three styles share one isinstance-based dispatch pattern.
    """


@dataclass(frozen=True)
class European(ExerciseStyle):
    pass  # exercisable only at expiry — no extra data needed


@dataclass(frozen=True)
class American(ExerciseStyle):
    pass  # exercisable any time up to expiry — no extra data needed, but engines
    # that support it must solve the free-boundary/optimal-stopping problem


@dataclass(frozen=True)
class Bermudan(ExerciseStyle):
    exercise_dates: tuple[float, ...]  # years from valuation date
