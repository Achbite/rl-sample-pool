# RL Sample Pool

[简体中文](README.md) | English

## 1. Run tests

Run inside the build container with the Contracts C++ bindings mounted:

```bash
bash ./test.sh
```

`test.sh` is the repository's unified test entrypoint and runs the current
development checks from an explicit allowlist.

## 2. Build the 0.14.0 artifact

Create the Contracts artifact first, then run the build from a clean Git savepoint:

```bash
(cd ../rl-contracts && bash build_artifact.sh)
bash build_artifact.sh
```

The script compiles the service and runs its tests. Output:

```text
../.workspace/artifacts/rl-sample-pool/0.14.0/<platform>/
```

## 3. Stage it into Learner

Do not copy the binary manually. Use the Learner sync entry point:

```bash
(cd ../rl-learner && bash scripts/sync_runtime_artifacts.sh)
```

## 4. Run it directly

The current backend stores independent processed transitions. A training
draw samples READY items uniformly without replacement; TRAINED ACK deletes
them, NACK restores READY, and capacity pressure evicts only the oldest READY
items in FIFO order. Behavior `model_step` is provenance rather than a stale
admission gate, and producer envelopes, GAE segments, and Learner batches remain
distinct units.

```bash
# When build/maze_sample_pool already exists
bash ./run.sh configs/pool_config.yaml
```

During full training the Learner container supervises this process; no separate launch is required.

## License

[MIT License](LICENSE)
