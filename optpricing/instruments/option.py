from dataclasses import dataclass, field

from optpricing.instruments.exercise import European, ExerciseStyle
from optpricing.payoffs.base import Payoff


@dataclass(frozen=True)
class Option:
    """Exercise style (when) + expiry (how long) + payoff (what).

    Deliberately no strike, barrier, or averaging-window fields here — those
    are payoff-specific and live on the Payoff itself, so Option has the same
    shape whether it wraps a vanilla call, an Asian, or a two-strike spread.
    """

    payoff: Payoff
    expiry: float  # years
    exercise: ExerciseStyle = field(default_factory=European)  # defaults to European, the common case
