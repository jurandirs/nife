"""The roots of NIFE: what every explainer and every explanation have, whatever their kind.

Every kind of explanation (`nife.counterfactual`, and those to come) derives its
explainers from `Explainer` and its explanations from `Explanation`. Every
explanation draws its plots the same way:

- each plot is a method named ``plot_<kind>``, which takes an optional matplotlib
  ``ax`` and returns the Axes it drew on;
- ``plot(kind)`` draws one of them by name, and ``plot()`` the default one
  (`default_plot`);
- `plot_kinds` lists the kinds available, with a one-line description.

Subclasses inherit the plots of their parents and add their own by defining more
``plot_<kind>`` methods. All of them share the look of `nife.plotting`.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Optional


class Explanation:
    """Base class of every explanation."""

    #: The kind drawn by ``plot()`` without a kind.
    default_plot: ClassVar[Optional[str]] = None

    @classmethod
    def plot_kinds(cls) -> Dict[str, str]:
        """The plots available, ``{kind: description}``: the parents' first, in the order they are defined."""
        kinds: Dict[str, str] = {}
        for klass in reversed(cls.__mro__):
            for name, member in vars(klass).items():
                if name.startswith("plot_") and name != "plot_kinds" and callable(member):
                    kinds.pop(name[5:], None)  # a subclass that redefines a plot keeps its place at the end
                    kinds[name[5:]] = (inspect.getdoc(member) or "").split("\n")[0]
        return kinds

    def plot(self, kind: Optional[str] = None, **kwargs):
        """Draw the plot `kind` of this explanation (the default plot without a kind).

        Keyword arguments go to the ``plot_<kind>`` method; see `plot_kinds` for the
        kinds. Returns the matplotlib Axes.
        """
        kind = self.default_plot if kind is None else kind
        kinds = self.plot_kinds()
        if kind is None:
            raise NotImplementedError(f"{type(self).__name__} has no default plot")
        if kind not in kinds:
            available = ", ".join(repr(k) for k in kinds) or "none"
            raise ValueError(f"unknown plot {kind!r} for {type(self).__name__}; available: {available}")
        return getattr(self, "plot_" + kind)(**kwargs)


class Explainer(ABC):
    """Base class of every explainer."""

    #: The class of the explanations returned by `explain`.
    explanation_type: ClassVar[type] = Explanation

    @abstractmethod
    def explain(self, x, **options) -> Explanation:
        """Explain the model's prediction for one instance."""
