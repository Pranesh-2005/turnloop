def best_scores(submissions):
    """Reduce a stream of (player, score) submissions to one entry per player.

    Each player's entry keeps their highest score across all of their
    submissions. The result is ordered by each player's most recent
    submission -- so a player who submitted early but submits again later
    moves to that later position, even if the later submission didn't beat
    their personal best.

    Raises ValueError on an empty list.
    """
    raise NotImplementedError
