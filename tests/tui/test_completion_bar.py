from cecli.tui.widgets.completion_bar import CompletionBar


def test_absolute_path_suggestions_stay_absolute():
    """Absolute suggestions are not converted to relative (../) paths."""
    bar = CompletionBar(
        suggestions=["/mnt/", "/srv/", "/etc/", "/dev/", "/opt/"],
        prefix="/workspace test /",
    )
    bar._compute_display_names()

    # The stored suggestions must remain absolute so selection inserts the correct path.
    assert bar.suggestions == ["/mnt/", "/srv/", "/etc/", "/dev/", "/opt/"]
    # The shared filesystem root is shown once as a prefix.
    assert bar._common_prefix == "/"
    assert bar._display_names == ["mnt/", "srv/", "etc/", "dev/", "opt/"]
    assert bar.current_selection == "/mnt/"


def test_relative_path_suggestions_kept():
    """Project-relative suggestions keep their existing display behavior."""
    bar = CompletionBar(
        suggestions=["src/main.py", "src/util.py", "tests/test.py"],
        prefix="/add ",
    )
    bar._compute_display_names()

    assert bar.suggestions == ["src/main.py", "src/util.py", "tests/test.py"]
    assert bar._display_names == ["src/main.py", "src/util.py", "tests/test.py"]
