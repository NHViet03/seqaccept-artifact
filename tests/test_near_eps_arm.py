from near_eps_arm import choose


def test_choose_puts_a_one_test_regression_just_above_eps():
    # 100 tests, recorded block 25; a candidate always failing 1 test fails a b-block w.p. b/100: b = 6 or 7 or 8 is near
    task = {"test_ids": [f"t{i}" for i in range(100)], "block_size": 25, "p0": 0.0}
    b, rates, near = choose(task, {"a": {"t1"}, "ok": set(), "order": None})
    assert near == 1 and 0.05 < rates["a"] <= 0.08 and b == 8
    assert rates["ok"] == 0.0 and "order" not in rates
