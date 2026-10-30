"""The roots of NIFE: the plots of every explanation, and the explainer."""

import pytest

from nife.base import Explainer, Explanation


class Parent(Explanation):
    default_plot = "first"

    def plot_first(self, scale=1, ax=None):
        """The first plot.

        More details.
        """
        return ("first", scale, ax)

    def plot_second(self, ax=None):
        """The second plot."""
        return ("second", ax)


class Child(Parent):
    def plot_third(self, ax=None):
        """A plot of the child only."""
        return ("third", ax)


class Redefines(Parent):
    def plot_first(self, scale=1, ax=None):
        """The first plot, redrawn."""
        return ("first again", scale, ax)


class NoDefault(Explanation):
    def plot_only(self):
        """The only plot."""
        return "only"


class Explains(Explainer):
    explanation_type = Child

    def explain(self, x, **options):
        return self.explanation_type()


def test_plot_kinds_list_the_plots_with_their_first_line():
    assert Parent.plot_kinds() == {"first": "The first plot.", "second": "The second plot."}
    assert Parent().plot_kinds() == Parent.plot_kinds()


def test_subclasses_inherit_and_add_plots():
    assert list(Child.plot_kinds()) == ["first", "second", "third"]
    assert Redefines.plot_kinds()["first"] == "The first plot, redrawn."
    assert Redefines().plot() == ("first again", 1, None)


def test_plot_draws_the_default_or_the_named_kind():
    child = Child()
    assert child.plot() == ("first", 1, None)
    assert child.plot("first", scale=2, ax="ax") == ("first", 2, "ax")
    assert child.plot("third") == ("third", None)
    assert child.plot(kind="second") == ("second", None)


def test_unknown_kind_lists_the_available_ones():
    with pytest.raises(ValueError, match="'first', 'second', 'third'"):
        Child().plot("fourth")


def test_without_a_default_plot():
    with pytest.raises(NotImplementedError):
        NoDefault().plot()
    assert NoDefault().plot("only") == "only"
    assert Explanation.plot_kinds() == {}


def test_explainers_return_their_explanation_type():
    explanation = Explains().explain(None)
    assert isinstance(explanation, Explanation) and explanation.plot("third") == ("third", None)


def test_explainer_is_abstract():
    with pytest.raises(TypeError):
        Explainer()
