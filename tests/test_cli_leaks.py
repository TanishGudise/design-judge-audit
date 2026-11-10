from dja import cli


def test_leak_check_matches_on_boundaries_not_spaces():
    banned = {"C1", "E4a", "b10", "5fa1633ca637ee11e343b641", "gde"}
    for text in ["label:\nC1\n", "(E4a)", "E1: x C1", "id b10.", "x 5fa1633ca637ee11e343b641", "gde/org"]:
        assert cli.leak_check_text(text, banned), text
    for text in ["b1008 is fine for token b10", "Rate it 10/10", "graphic design", "C10 apples"]:
        hit = cli.leak_check_text(text, banned)
        assert hit is None or text.startswith("b1008"), (text, hit)


def test_leak_check_catches_decimal_scores():
    assert cli.leak_check_text("human mean 7.12", set()) == "7.1"
    assert cli.leak_check_text("from 1 (very poor) to 10 (excellent)", set()) is None
