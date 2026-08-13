# RL Sample Pool

[简体中文](README.md) | English

## 1. Build the 0.11.0 artifact

Create the Contracts artifact first, then run the build from a clean Git savepoint:

```bash
(cd ../rl-contracts && bash build_artifact.sh)
bash build_artifact.sh
```

The script compiles the service and runs its tests. Output:

```text
../.workspace/artifacts/rl-sample-pool/0.11.0/<platform>/
```

## 2. Stage it into Learner

Do not copy the binary manually. Use the Learner sync entry point:

```bash
(cd ../rl-learner && bash scripts/sync_runtime_artifacts.sh)
```

## 3. Run it directly

```bash
./maze_sample_distributor configs/distributor_config.yaml
```

During full training the Learner container supervises this process; no separate launch is required.

## License

[MIT License](LICENSE)
