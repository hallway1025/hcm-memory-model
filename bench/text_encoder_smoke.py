"""Small, dependency-free regression for CharacterSemanticEncoder."""
import sys
sys.path.insert(0, "src")

import numpy as np
from hcm import CharacterSemanticEncoder, HCM, Config

enc = CharacterSemanticEncoder(dim=32, window=2, seed=17).fit([
    "红苹果", "青苹果", "红色水果", "青色水果", "😀",
])
X = enc.transform(["红苹果", "未知😀", ""])
assert X.shape == (3, 32)
assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)
assert not np.allclose(X[1], X[2])
np.testing.assert_allclose(enc.encode("😀"), enc.encode("😀"))

m = HCM(Config(d_x=32, d_c=32, capacity=8, beta=100.0, L=0,
              use_resid=False, lam=1.0, seed=0))
for x in X:
    m.observe(x)
assert m.E.n == 3
assert m.query(X[1], do_rif=False)["idx"] >= 0
print("text_encoder_smoke: PASS", "vocab=%d" % len(enc.vocab), "episodes=%d" % m.E.n)
