# RL Sample Pool

[简体中文](README.md) | English

## 1. Run tests

Run inside the build container with the Contracts C++ bindings mounted:

```bash
bash ./test.sh
```

`test.sh` is the repository's unified test entrypoint and runs the current
development checks from an explicit allowlist.

## 2. Build the 0.15.0 artifact

Create the task-neutral training-contract artifact first, then build from the
current source:

```bash
(cd ../rl-contracts && bash build_artifact.sh training)
bash build_artifact.sh
```

The artifact script only compiles the production binary. Tests remain an explicit
`bash ./test.sh` operation from section 1. Output:

```text
../.workspace/artifacts/rl-sample-pool/0.15.0/<platform>/
```

## 3. Stage it into Learner

Prefer the Learner sync entrypoint to copy the complete artifact:

```bash
(cd ../rl-learner && bash scripts/sync_runtime_artifacts.sh)
```

For development artifacts, Learner's `make deps` builds and stages both required
binaries. `make shell` never performs that synchronization implicitly.

## 4. Run it directly

The current backend stores independent processed transitions. A training
draw samples READY items uniformly without replacement; TRAINED ACK deletes
them, NACK restores READY, and capacity pressure evicts only the oldest READY
items in FIFO order. Behavior `model_step` is provenance rather than a stale
admission gate, and producer envelopes, GAE segments, and Learner batches remain
distinct units. `action_mask` is stored and delivered unchanged like every other
sample field. Sample Pool does not interpret task action semantics or require masks
to be enabled.

```bash
# When build/maze_sample_pool already exists
bash ./run.sh configs/pool_config.yaml
```

During full training the Learner container supervises this process; no separate launch is required.

## License

[MIT License](LICENSE)
