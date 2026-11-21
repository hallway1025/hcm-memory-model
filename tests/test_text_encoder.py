import sys
sys.path.insert(0, "src")

import numpy as np

from hcm import CharacterSemanticEncoder


def test_fit_transform_is_normalized_and_deterministic():
    messages = ["红苹果", "青苹果", "红色水果", "青色水果"]
    enc = CharacterSemanticEncoder(dim=8, window=2, seed=17).fit(messages)
    out = enc.transform(messages)
    assert out.shape == (4, 8)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(enc.encode("😀"), enc.encode("😀"))
    np.testing.assert_allclose(enc.encode(""), enc.encode(""))


def test_oov_and_empty_do_not_collapse_to_same_vector():
    enc = CharacterSemanticEncoder(dim=16, seed=4).fit(["甲乙", "乙丙"])
    assert float(enc.encode("😀") @ enc.encode("😃")) < 0.95
    assert not np.allclose(enc.encode(""), enc.encode("😀"))


def test_normalize_removes_markers_and_whitespace():
    enc = CharacterSemanticEncoder(dim=4).fit(["甲乙"])
    assert enc.normalize(" 甲[微笑]\n乙 ") == "甲乙"


def test_marker_identity_empty_batch_and_singleton():
    enc = CharacterSemanticEncoder(dim=32, seed=4).fit(["甲"])
    assert enc.transform([]).shape == (0, 32)
    assert not np.allclose(enc.encode("[微笑]"), enc.encode("[哭]"))
    assert not np.allclose(enc.encode("[微笑]"), enc.encode(""))
    np.testing.assert_allclose(enc.encode(" [微笑] "), enc.encode("[微笑]"))
    np.testing.assert_allclose(np.linalg.norm(enc.encode("甲")), 1, atol=1e-5)
    other = CharacterSemanticEncoder(dim=32, seed=4).fit([])
    np.testing.assert_allclose(enc.encode("[微笑]"), other.encode("[微笑]"))


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print("text encoder: %d tests PASS" % len(tests))
