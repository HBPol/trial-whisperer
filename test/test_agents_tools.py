from app.agents.tools import check_eligibility


def test_check_eligibility_age_within_range():
    criteria = {"inclusion": ["Age 18 to 65"], "exclusion": []}
    result = check_eligibility(criteria, {"age": 35})

    assert result["eligible"] is True
    assert result["reasons"] == []


def test_check_eligibility_age_outside_range():
    criteria = {"inclusion": ["Age 18 to 65"], "exclusion": []}
    result = check_eligibility(criteria, {"age": 70})

    assert result["eligible"] is False
    assert any("outside required range" in reason for reason in result["reasons"])


def test_check_eligibility_missing_age():
    criteria = {"inclusion": ["Age 18 to 65"], "exclusion": []}
    result = check_eligibility(criteria, {})

    assert result["eligible"] is False
    assert any("Missing age information" in reason for reason in result["reasons"])


def test_check_eligibility_sex_only_rule_allows_matching_patient():
    criteria = {"inclusion": ["Female participants only"], "exclusion": []}
    result = check_eligibility(criteria, {"sex": "Female"})

    assert result["eligible"] is True
    assert result["reasons"] == []


def test_check_eligibility_sex_only_rule_blocks_mismatch():
    criteria = {"inclusion": ["Female participants only"], "exclusion": []}
    result = check_eligibility(criteria, {"sex": "Male"})

    assert result["eligible"] is False
    assert any("not permitted" in reason for reason in result["reasons"])


def test_check_eligibility_exclusion_age_range_blocks_patient():
    criteria = {"inclusion": [], "exclusion": ["Age 30 to 40"]}
    result = check_eligibility(criteria, {"age": 35})

    assert result["eligible"] is False
    assert any("exclusion criterion" in reason for reason in result["reasons"])
