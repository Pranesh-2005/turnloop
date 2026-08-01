import pytest
from leaderboard import best_scores


def test_single_submission():
    assert best_scores([("ana", 10)]) == [("ana", 10)]


def test_keeps_the_higher_score():
    assert best_scores([("ana", 10), ("ana", 25)]) == [("ana", 25)]


def test_orders_by_most_recent_submission():
    # "ana" submits first, then "bo", then "ana" again -- "ana" moves to
    # the position of her second (most recent) submission.
    assert best_scores([("ana", 10), ("bo", 25), ("ana", 15)]) == [("bo", 25), ("ana", 15)]


def test_a_later_lower_submission_still_moves_the_player():
    # "ana"'s second submission is *worse* than her first, so her kept
    # score is still 30 -- but her position still follows the later
    # submission, independent of which one actually won.
    assert best_scores([("ana", 30), ("bo", 5), ("ana", 2)]) == [("bo", 5), ("ana", 30)]


def test_empty_raises():
    with pytest.raises(ValueError):
        best_scores([])
