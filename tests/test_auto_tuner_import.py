def test_auto_tuner_import_and_config():
    import auto_tuner  # should import without NameError

    # TuneConfig should be available and instantiable
    cfg = auto_tuner.TuneConfig(cycles=0, episodes_per_cycle=10)
    assert cfg.cycles == 0
    assert cfg.episodes_per_cycle == 10

