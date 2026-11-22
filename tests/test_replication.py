from mist.replication import calc_class_weight, lr_at


def test_class_weights_match_paper_formula_on_table1_counts():
    # AttnSleep paper Table I, Sleep-EDF-20 epoch counts (W, N1, N2, N3, REM)
    w = calc_class_weight([8285, 2804, 17799, 5703, 7717])
    assert w == [0.3, 0.72, 0.3, 0.2, 0.3]


def test_lr_schedule():
    assert lr_at(0) == lr_at(9) == 1e-3 and lr_at(10) == lr_at(99) == 1e-4
