from chatv1.health import check_health


def test_health_reports_missing_checkpoint(tmp_path):
    report = check_health(str(tmp_path / "artifacts" / "model.pt"))
    assert report.status == "degraded"
    assert report.checkpoint_present is False
