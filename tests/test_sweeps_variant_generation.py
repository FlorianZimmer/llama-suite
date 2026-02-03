from __future__ import annotations

from llama_suite.webui.api.sweeps import SweepDimension, SweepRange, generate_variants


def test_generate_variants_cartesian_product() -> None:
    dims = [
        SweepDimension(path=["cmd", "ctx-size"], value_type="int", values=[8192, 16384]),
        SweepDimension(path=["cmd", "parallel"], value_type="int", range=SweepRange(start=1, end=2, step=1)),
    ]
    variants = generate_variants(dims)
    assert len(variants) == 4

    got = {(v["cmd.ctx-size"], v["cmd.parallel"]) for v in variants}
    assert got == {(8192, 1), (8192, 2), (16384, 1), (16384, 2)}

