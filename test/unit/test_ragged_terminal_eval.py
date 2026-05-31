import numpy as np

from poker_ai.research.ragged_terminal_eval import pad_ragged_2d_arrays


def test_pad_ragged_2d_arrays_preserves_rows_and_mask():
    first = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    second = np.asarray([[5.0, 6.0]], dtype=np.float32)

    padded, mask = pad_ragged_2d_arrays([first, second], pad_value=-1.0)

    assert padded.shape == (2, 2, 2)
    assert mask.tolist() == [[True, True], [True, False]]
    np.testing.assert_allclose(padded[0], first)
    np.testing.assert_allclose(padded[1, 0], second[0])
    np.testing.assert_allclose(padded[1, 1], [-1.0, -1.0])
