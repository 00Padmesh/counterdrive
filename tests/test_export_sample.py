from counterdrive.export_sample import select_index


def test_sample_selection_is_deterministic() -> None:
    assert select_index(10, 3, False, 42) == 3
    assert select_index(10, 0, True, 42) == select_index(10, 0, True, 42)
