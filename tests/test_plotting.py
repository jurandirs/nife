"""The look shared by NIFE's plots."""

import matplotlib as mpl
import matplotlib.pyplot as plt

from nife import plotting


def test_style_does_not_change_the_settings():
    before = mpl.rcParams["axes.spines.top"], mpl.rcParams["font.size"]
    with plotting.style():
        assert mpl.rcParams["axes.spines.top"] is False
    assert (mpl.rcParams["axes.spines.top"], mpl.rcParams["font.size"]) == before


def test_style_keeps_what_else_changes_inside():
    before = mpl.rcParams["interactive"]
    with plotting.style():
        mpl.rcParams["interactive"] = not before  # as Jupyter does when it draws the first figure
    assert mpl.rcParams["interactive"] is (not before)
    mpl.rcParams["interactive"] = before


def test_styled_functions_draw_inside_the_style():
    @plotting.styled
    def draw():
        return mpl.rcParams["axes.titlelocation"]

    assert draw() == "left"


def test_new_axes_reuses_the_given_axes():
    _, ax = plt.subplots()
    assert plotting.new_axes(ax) is ax
    new = plotting.new_axes(figsize=(3, 2))
    assert new is not ax
    assert tuple(new.figure.get_size_inches()) == (3, 2)


def test_titles_and_legend():
    ax = plotting.new_axes()
    plotting.set_titles(ax, "Title", "first part · second part · third part")
    assert ax.get_title(loc="left") == "Title"
    assert ax.texts[-1].get_text().startswith("first part")
    legend = plotting.legend_below(ax, [plotting.dot("red"), plotting.swatch("blue")], ["a", "b"])
    assert [t.get_text() for t in legend.get_texts()] == ["a", "b"]


def test_wrap_parts_breaks_at_the_separators():
    text = "aaaa bbbb · cccc dddd · eeee"
    assert plotting.wrap_parts(text, 12) == "aaaa bbbb\ncccc dddd\neeee"
    assert plotting.wrap_parts(text, 100) == text


def test_formats():
    assert plotting.format_value(3.0) == "3"
    assert plotting.format_value(0.0123456) == "0.01235"
    assert plotting.format_value("red") == "red"
