import numpy as np
from amsrr_det.contact_increment import velocity_response


def test_increment_matches_full_equations_with_compliant_contacts():
    rng = np.random.default_rng(47)
    a = rng.normal(size=(12, 12))
    e = a.T @ a + np.eye(12)
    b = rng.normal(size=(12, 5))
    c = rng.normal(size=(7, 12))
    d = np.diag(np.geomspace(1e-6, 1., 7))
    full = np.block([[e, -c.T], [c, d]])
    expected = np.linalg.solve(full, np.vstack([b, np.zeros((7, 5))]))[:12]
    np.testing.assert_allclose(velocity_response(e, b, c, d), expected,
                               rtol=1e-10, atol=1e-12)


def test_redundant_support_preserves_velocity_without_free_motion_penalty():
    rng = np.random.default_rng(48)
    a = rng.normal(size=(10, 10))
    e = a.T @ a + np.eye(10)
    b = rng.normal(size=(10, 4))
    independent = rng.normal(size=(3, 10))
    c = np.vstack([independent, independent[0], 2 * independent[1]])
    response = velocity_response(e, b, c, np.zeros((5, 5)))
    full = np.block([[e, -c.T], [c, np.zeros((5, 5))]])
    expected = np.linalg.lstsq(full, np.vstack([b, np.zeros((5, 4))]), rcond=1e-12)[0][:10]
    np.testing.assert_allclose(response, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(c @ response, 0., atol=1e-12)
    assert np.linalg.norm(c @ np.linalg.solve(e, b)) > 1.
