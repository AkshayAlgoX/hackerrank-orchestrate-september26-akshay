import os
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "evaluation"))


import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_default_evidence_cache(tmp_path_factory):
    """Point gather_evidence's default cache at a per-test temp file.

    The default is code/evidence_cache.json, an untracked file that holds a
    developer's model extractions. Tests must neither depend on it nor rewrite
    it. Tests that pass an explicit cache_path are unaffected. Session scope so
    module-scoped fixtures that call run() are covered too.
    """
    from buyorwait.extraction import gather

    defaults = list(gather.gather_evidence.__defaults__)
    idx = gather.gather_evidence.__code__.co_varnames.index("cache_path") - (
        gather.gather_evidence.__code__.co_argcount - len(defaults)
    )
    assert defaults[idx] == gather.DEFAULT_CACHE, "gather_evidence signature changed; update this fixture"
    defaults[idx] = str(tmp_path_factory.mktemp("evidence") / "default-evidence-cache.json")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gather.gather_evidence, "__defaults__", tuple(defaults))
        yield
