import unittest

import numpy as np

from entrance_anchor import EntranceAnchorConstraint, apply_entrance_anchor


class DemonstratorAnchorPipelineTests(unittest.TestCase):
    def test_stage_is_noop_when_disabled_and_translates_when_enabled(self):
        curve = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

        disabled = apply_entrance_anchor(
            EntranceAnchorConstraint(enabled=False),
            curve,
            166.8,
        )
        enabled_state = EntranceAnchorConstraint(enabled=True)
        apply_entrance_anchor(enabled_state, curve, 156.8)
        enabled = apply_entrance_anchor(enabled_state, curve, 166.8)

        np.testing.assert_allclose(disabled, curve)
        np.testing.assert_allclose(enabled[:, 0], [0.01, 0.11])
        np.testing.assert_allclose(enabled[:, 1:], 0.0)


if __name__ == "__main__":
    unittest.main()
