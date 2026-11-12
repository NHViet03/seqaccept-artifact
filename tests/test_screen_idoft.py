from screen_idoft import shortlist

HEADER = "Project URL,SHA Detected,Pytest Test Name (PathToFile::TestClass::TestMethod or PathToFile::TestMethod),Category,Status,PR Link,Notes\n"


def csv_text(*rows):
    return HEADER + "".join(",".join(r) + "\n" for r in rows)


def row(url, cat, status="", sha="abc", test="t.py::test_x"):
    return [url, sha, test, cat, status, "", ""]


def test_keeps_order_dependent_and_random_flaky_tests_only():
    text = csv_text(
        row("https://github.com/a/od", "OD-Vic"),
        row("https://github.com/a/nod", "NOD"),
        row("https://github.com/a/nio", "NIO"),  # fails only when run twice in one process: fresh containers make it irrelevant
    )
    assert [r["url"] for r in shortlist(text)] == ["https://github.com/a/od", "https://github.com/a/nod"]


def test_skips_dead_projects():
    text = csv_text(
        row("https://github.com/a/alive", "OD"),
        row("https://github.com/a/old", "OD", "Unmaintained"),
        row("https://github.com/a/arch", "OD", "RepoArchived"),
        row("https://github.com/a/gone", "OD", "RepoDeleted"),
    )
    assert [r["url"] for r in shortlist(text)] == ["https://github.com/a/alive"]


def test_groups_tests_per_repo_and_ranks_repos_with_both_kinds_of_flakiness_first():
    text = csv_text(
        row("https://github.com/a/od_only", "OD-Vic", test="t.py::a"),
        row("https://github.com/a/od_only", "OD-Brit", test="t.py::b"),
        row("https://github.com/a/mixed", "OD-Vic", test="t.py::c"),
        row("https://github.com/a/mixed", "NOD", test="t.py::d"),
    )
    rows = shortlist(text)
    assert rows[0]["url"] == "https://github.com/a/mixed"
    assert rows[1]["tests"] == ["t.py::a", "t.py::b"] and rows[1]["sha"] == "abc"
