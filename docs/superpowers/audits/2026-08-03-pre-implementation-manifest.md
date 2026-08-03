# Pre-implementation Worktree Manifest

Captured before implementation edits on 2026-08-03. This manifest records the
union of paths in HEAD, the index, and the untracked working tree. It preserves
the mixed index/worktree state that predates the repository-consistency work.

## Approved implementation allowlist

```text
README.md
docs/superpowers/audits/2026-08-03-pre-implementation-manifest.md
docs/superpowers/plans/2026-08-03-repository-consistency-readme.md
findings.md
progress.md
scripts/validate_repository_consistency.py
task_plan.md
src/rice_phenology_hypernet/types.py
src/rice_phenology_hypernet/data/__init__.py
src/rice_phenology_hypernet/data/io.py
src/rice_phenology_hypernet/experiments/__init__.py
src/rice_phenology_hypernet/experiments/dvr_core.py
src/rice_phenology_hypernet/experiments/runner_dvr.py
src/rice_phenology_hypernet/experiments/threshold_utils.py
src/rice_phenology_hypernet/experiments/regional_grid_projection.py
src/rice_phenology_hypernet/experiments/regional_grid_analysis.py
src/rice_phenology_hypernet/models/__init__.py
src/rice_phenology_hypernet/models/dvr_objective.py
src/rice_phenology_hypernet/models/m1_dvr_con.py
```

## Snapshot

```text
branch | main
upstream | origin/main
ahead-behind | ahead=6 behind=0
path | HEAD blob | index blob | porcelain status | worktree SHA-256
.gitignore | 15958ab2fffbc6f676be0e7207ec9a5794e9a381 | 15958ab2fffbc6f676be0e7207ec9a5794e9a381 | CLEAN | 67d52b7ed825a57b250eb7d3b7025e23480477f1e06013ca375aef56006e7b95
README.md | e87e4075f317f3d54574a606909dfded0d1d5751 | e87e4075f317f3d54574a606909dfded0d1d5751 |  M | c8d5c648be352fdd1640091065c7abdf23aa6a16a552c077d2c27852fb734d0e
artifacts/.gitkeep | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | CLEAN | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
artifacts/README.md | 45e22479bcd6b61bf188f2df36627f38a458a274 | ABSENT | D  | ABSENT
artifacts/eval/.gitkeep | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | CLEAN | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
artifacts/features/.gitkeep | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | CLEAN | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
artifacts/figures/.gitkeep | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | CLEAN | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
artifacts/models/.gitkeep | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | CLEAN | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
artifacts/tables/.gitkeep | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | CLEAN | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
data/README.md | 24bc1f0f966ae881faafd07fc6b7e97913613384 | ABSENT | D  | ABSENT
data/artifacts/.gitkeep | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | CLEAN | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
data/artifacts/features/.gitkeep | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | ABSENT | D  | ABSENT
data/boundary/.gitkeep | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | CLEAN | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
data/processed/.gitkeep | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | CLEAN | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
data/raw/.gitkeep | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 | CLEAN | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
docs/superpowers/plans/2026-08-03-config-derived-model-parameters.md | ABSENT | c798dd0cbd31be726024f152089df659919386a7 | AD | ABSENT
docs/superpowers/plans/2026-08-03-repository-consistency-readme.md | ABSENT | ABSENT | ?? | 5f1ed2833687e793e86866fd94747a19849b3786e8333962a0320d861bd435e9
docs/superpowers/specs/2026-08-03-config-derived-model-parameters-design.md | c93e0326067388ef617677d0fde70e7fdf8eeec9 | c93e0326067388ef617677d0fde70e7fdf8eeec9 |  D | ABSENT
docs/superpowers/specs/2026-08-03-readme-project-overview-design.md | 9cb25d1e36af7c4ea2fa2f28425b710604558669 | 9cb25d1e36af7c4ea2fa2f28425b710604558669 |  D | ABSENT
docs/superpowers/specs/2026-08-03-repository-consistency-readme-design.md | ebbd7ec742abfe538e926345a06a9becd658ab39 | ebbd7ec742abfe538e926345a06a9becd658ab39 | CLEAN | 74278322d3a5f195386a15eb70efa5c17ddde728ad1882b5fc8afaf884344d85
findings.md | ABSENT | 140b3f252fc240bc18d21a38984789f09a573aa2 | AM | 291bce3aca168d165be160e668801b5bc879257936e449416bd74e8c796e0499
progress.md | ABSENT | 08ac1fefa66284755fc81aa736b5d7b7f3872cb8 | AM | d86347daa4aa13eff01765db3d5d75cc461e5eae629797dbc4f4da43e8d8ae0e
pyproject.toml | d7f7f2c512947de88548089d5f8e147c42a71b52 | ABSENT | D  | ABSENT
requirements.txt | 0ab5b574e0cd76bb35163d3206a8200d0d88aaf1 | 0ab5b574e0cd76bb35163d3206a8200d0d88aaf1 | CLEAN | 2535b2c98bf3c76aaa1d3972e8803efb744e73b8e665abd76b31223fb43d7023
scripts/china_rice_calendar/coarsen_middle_rice_rasters.py | 18613206844c026adf5261bdbfad96c8f46c3850 | 18613206844c026adf5261bdbfad96c8f46c3850 | CLEAN | 9a949f88b266ced83359ea57c0f6d87255f22c077729139a6249b7b0ef9f6b4e
scripts/china_rice_calendar/download_rice_calendar.py | 7439a3799591331472d0dd72acaf1a3669ef6ec8 | 7439a3799591331472d0dd72acaf1a3669ef6ec8 | CLEAN | 3ff98de4d73c07788d80a9cf293208630453a0372ca2bd316538043bf9f53579
scripts/china_rice_calendar/extract_middle_rice_pixels.py | ee55cc8b336a45ae4a460d45aef2f80fcbbb885a | ee55cc8b336a45ae4a460d45aef2f80fcbbb885a | CLEAN | 96c9e4e5b5b758bb5c6f88de6934869f3ee642353b06548a3f4af6283a6cab19
scripts/china_rice_calendar/visualize_middle_rice.py | 46eddaa48e207a8afb964c8c22792b8b612e4fc7 | 46eddaa48e207a8afb964c8c22792b8b612e4fc7 | CLEAN | 3d661c9bcaa12029da01a1ff584fa96ba39c4938f21bc0151ccd95c8eaf01df6
scripts/meteo_download/download_regional_grid_weather_gee.py | 142d15acf0728dd2b6d65a44a47795427ef9787b | 142d15acf0728dd2b6d65a44a47795427ef9787b | CLEAN | 5b846739e9e255f6567319a3bcd16a20f680cc9fb957831535b0935eb7299e05
scripts/meteo_download/download_site_wether.py | 943ab2fad6bf63baf97049a69ceee7ec41d6a8e3 | ABSENT | D  | ABSENT
scripts/meteo_download/standardize_regional_grid_weather_gee.py | 5706e583f4774558b4765455d9bf5b4e5bbbb526 | 5706e583f4774558b4765455d9bf5b4e5bbbb526 | CLEAN | 9a2ea1f8871491d3ee0c07d1f84c637280c0194fddf356271378309a23ff1259
src/rice_phenology_hypernet/__init__.py | e8641c8ad938544a0a9f3d2859282d9f8f92c5ac | e8641c8ad938544a0a9f3d2859282d9f8f92c5ac | CLEAN | c082c8a5936bbe76fdebaa2b6f5000a855a60ba62607d700e1170356f40fe79b
src/rice_phenology_hypernet/cli.py | 01606d76a2b68d2b81ae6f8ae522029f1daca7cc | ABSENT | D  | ABSENT
src/rice_phenology_hypernet/config.py | 8c0f8d661bb6f7cf6e11d5d5348faec043164b9b | ABSENT | D  | ABSENT
src/rice_phenology_hypernet/data/__init__.py | c8015c048c24d4983bde9b2a365f56b544ef38b8 | c8015c048c24d4983bde9b2a365f56b544ef38b8 | CLEAN | b1491aa5f3657ad8270dc0cc7380221d98c7e22f8bef71a88ee0291a0125220e
src/rice_phenology_hypernet/data/dataset_dvr.py | 86e6f1510aa325c8029701f92f1732ebba3435d5 | ABSENT | D  | ABSENT
src/rice_phenology_hypernet/data/daylength.py | 1be49e3252a039d6b1febcfbf40042b213f4b290 | 1be49e3252a039d6b1febcfbf40042b213f4b290 | CLEAN | 6987d082ca087c407e3c2d8c1930fa1c58e2fcd37488991fc34845447c9c058d
src/rice_phenology_hypernet/data/io.py | 980645f6aa0e37c1f3d3af5dead44dcc8be1c7fd | 980645f6aa0e37c1f3d3af5dead44dcc8be1c7fd | CLEAN | 1a91b50ac277143ca037478ce2738c44068431f8d2ee79097db6ca71c58623b6
src/rice_phenology_hypernet/evaluation/__init__.py | cc45dc755858a88eaea503fbd0c57e068dc93fee | cc45dc755858a88eaea503fbd0c57e068dc93fee | CLEAN | 6c4a0e5b53bbb2364e24e79df56b8015293892df825c0d696316646332435bcd
src/rice_phenology_hypernet/evaluation/metrics.py | 3f80ee0269c821f6bcbebdef09573bea981c1ae4 | 3f80ee0269c821f6bcbebdef09573bea981c1ae4 | CLEAN | afc585991fab8d28a5688964e9597602a3da5be0e3459f42893714c0a2670d21
src/rice_phenology_hypernet/experiments/__init__.py | 1b85109aa9916cdabf94f153ed4fdebe79d44977 | 1b85109aa9916cdabf94f153ed4fdebe79d44977 | CLEAN | 67bc050576d56d9ee796c4b393f5e71d76aee45448e8e21710a9395d73abaeed
src/rice_phenology_hypernet/experiments/dvr_summary.py | ffbeb983d9e7d9d6c69d2d4b4ae340a4472bfd23 | ffbeb983d9e7d9d6c69d2d4b4ae340a4472bfd23 | CLEAN | 9a13f794d33a9a61904499fda2881a81da8ff8d3ffaed3b7a880e280c74ed08c
src/rice_phenology_hypernet/experiments/modifier_interpretability.py | a0522c8c206ad4d291556e3cfee0767ec4ad061f | ABSENT | D  | ABSENT
src/rice_phenology_hypernet/experiments/regional_grid_analysis.py | 70928c3b0bb5f6bd019e694706b39756f2167a02 | 70928c3b0bb5f6bd019e694706b39756f2167a02 | CLEAN | d362f01955959976fb9e0a1a3fc651830534648ae45ae5c0a313cbf9c297fcf0
src/rice_phenology_hypernet/experiments/regional_grid_projection.py | 1532a2cdec6189e43baa571d3da6e3d36e35702c | 1532a2cdec6189e43baa571d3da6e3d36e35702c | CLEAN | 0d5828c762e9fc13603a7575ac36ac2be51762ff1dbaba875ec0f5b7b6f102b9
src/rice_phenology_hypernet/experiments/regional_reviving_offset_sensitivity.py | 4420e2c1c646c60d5504d7da9c1183b42a8ebf21 | ABSENT | D  | ABSENT
src/rice_phenology_hypernet/experiments/runner_dvr.py | 52d763f29eb5daadaa0c1afb78931f82e94e279c | 5318eddcf18b4a23aa6147413a29fca76dd5591e | MM | a0c0c37188b0f9ba26f38f262d305e9d32b6e3dbf4bd6892d8261c77e73329d5
src/rice_phenology_hypernet/experiments/splits.py | 49418a4862faa864c98acbdc13ab3f9d2855b989 | ABSENT | D  | ABSENT
src/rice_phenology_hypernet/experiments/threshold_utils.py | 1a19d0f7f1e88573a62b243486d9e9ac993c4fcf | 1a19d0f7f1e88573a62b243486d9e9ac993c4fcf | CLEAN | 95447d44eeccd838d2873e5f3c41d0099bf10080d95a5bc01a021e426cbfa696
src/rice_phenology_hypernet/features/__init__.py | 41224883471855dc50724d166cb8cc076c862feb | ABSENT | D  | ABSENT
src/rice_phenology_hypernet/features/engineering.py | f830027c63e25e9d7c5b9e9eb1363fc88fb59c71 | ABSENT | D  | ABSENT
src/rice_phenology_hypernet/figures/__init__.py | 42776499c7d5d3455e95cf78bc2b326b40bc1d17 | ABSENT | D  | ABSENT
src/rice_phenology_hypernet/figures/builder.py | 0573089ac846e2d50ad90025d125a993b965a7dd | ABSENT | D  | ABSENT
src/rice_phenology_hypernet/models/__init__.py | 75cc1d287b29775e76f1de213c5574d376e536b9 | 75cc1d287b29775e76f1de213c5574d376e536b9 | CLEAN | 4cac6e206e55d1df9a23d5be630af3aca8b073652d0e05e04845aef7c969068a
src/rice_phenology_hypernet/models/dvr_loss.py | 1d93b76393943580a5156764b505f7901fcfe826 | ABSENT | D  | ABSENT
src/rice_phenology_hypernet/models/m0.py | 13ce7108722a9ef2d5bf99cf7ab0d1610a7d63ce | 78c44ec7f64a0b7d59a2e49f74ca4ab669d17c92 | M  | c0f419025c05a6b93d45a19e17cd0a13c1e0045c5a2eddc0ff03196076b56d1c
src/rice_phenology_hypernet/models/m1_dvr_con.py | fdb07f242e998fc9b6f090a7033ad002034e715e | f55aba785f2b74af292a83e3b824ba9cb96822b7 | MM | 75e9748401269024ca2e67a5a4dba9881521e853e2dc2fda0138dc304333c9c9
src/rice_phenology_hypernet/models/m1_v2_dvr.py | 85970263b3433ad12d876c85a2727c15cfe3a957 | 85970263b3433ad12d876c85a2727c15cfe3a957 |  M | 48b68d83958c613fe8f33c32a47662ce3f77af223562b8a42b023df530cde571
src/rice_phenology_hypernet/models/physics.py | fabbe3cce3261ad54749192529356aa96619c89d | fabbe3cce3261ad54749192529356aa96619c89d | CLEAN | 35a7031d4c450d6b3011bfc1c4b7396858a7559e7225692c7554dcdccf8dcadf
src/rice_phenology_hypernet/runtime.py | b5dc3f2f17cf19ab48da7b7e4f6bc3d64e94c2d8 | b5dc3f2f17cf19ab48da7b7e4f6bc3d64e94c2d8 | CLEAN | 046f0cca9c56d64a5ce870e626201355c12e66bdb36adde3598d41f76db3b9ee
src/rice_phenology_hypernet/settings.py | f2f825ea2c653bd1b14590768a2b197ecf4f5f63 | f2f825ea2c653bd1b14590768a2b197ecf4f5f63 | CLEAN | 96d72059e0a03e9712d402c2ed5d227fb96716e63095a322a50e0e87e60359d8
src/rice_phenology_hypernet/tables/__init__.py | fe3f139dc2f33e300bd70017e54432f587727fa7 | ABSENT | D  | ABSENT
src/rice_phenology_hypernet/tables/builder.py | 6cd9b0756544b99253b6e4de1b0387d55953c330 | ABSENT | D  | ABSENT
src/rice_phenology_hypernet/types.py | 09e3194ded53ea778d5a1e8a764a9accffaccf27 | 09e3194ded53ea778d5a1e8a764a9accffaccf27 | CLEAN | 1b26c0b62ca6e43d2f07c4ec7c6752c1b236df03bcec30142da53b1028760239
task_plan.md | ABSENT | cc4293de2ae10f748d9ef2f8f3f98ea5a334850d | AM | b2b02658d6a8da21a5f5c699bb5cf51f9baf7d3b1e0db55249ff747c911a8daf
tests/conftest.py | e7a624a284b71fd2a8424a31bf171dafa523e89d | ABSENT | D  | ABSENT
tests/test_config_loading.py | b263bbbebe22be9b0f1514a7716e54d77bbe8a48 | ABSENT | D  | ABSENT
tests/test_data_pipeline.py | 2c498635de1eb3704d91d2199e543e796beac9bf | ABSENT | D  | ABSENT
tests/test_deployment_materialization.py | 495d98f381d0d2e75ec63dbac5e7bfb387c45878 | ABSENT | D  | ABSENT
tests/test_metrics.py | 315967be95866c854896a9a95fade521a8977fb8 | ABSENT | D  | ABSENT
tests/test_models.py | 008e65dd73ec86772498f22d5b8362a87740d44e | ABSENT | D  | ABSENT
tests/test_modifier_interpretability.py | 451b1ff2235a59db217dd8dbfb70a2f0ea0ffb75 | ABSENT | D  | ABSENT
tests/test_regional_grid_projection.py | 93d96365c5de2058248e594cfad8f92a260a2622 | ABSENT | D  | ABSENT
tests/test_run_outputs.py | 7baf4173b25fe3a7f55bd0bb6cd440d42a72f1fa | ABSENT | D  | ABSENT
```
