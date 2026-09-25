# RL Sample Pool

[简体中文](README.md) | English

## 1. Run tests

Run inside the build container with the Contracts C++ bindings mounted:

```bash
bash ./test.sh
```

`test.sh` is the repository's unified test entrypoint and runs the current
development checks from an explicit allowlist.

## 2. Build the current artifact

Supply an existing Contracts Training bundle containing `cpp/` and `sdk/`:

```bash
CONTRACT_CPP_DIR=/path/to/training/cpp bash ./build.sh
```

This repository builds only its own source. It never builds or synchronizes upstream dependencies.
Tests run explicitly through `bash ./test.sh`. The default output is
`build/artifacts/production/<platform>/`; set `RL_ARTIFACT_ROOT` to select another output root.
`build_dev_artifact.sh` uses the same implementation with the default output
`build/artifacts/development/<platform>/current/`.

## 3. Stage it into Learner

With Framework's sibling repository layout, run from `rl-framework`:

```bash
bash ./sync_artifacts.sh
# Development artifact channel
bash ./sync_artifacts.sh --development
```

Framework calls the Contracts, Pool and Distributor build entrypoints and explicitly stages
selected artifacts into Learner. Component builds do not require this workspace layout.

## 4. Run it directly

The current backend stores independent processed transitions. A training
draw samples READY items uniformly without replacement; TRAINED ACK deletes
them, NACK restores READY, and capacity pressure evicts only the oldest READY
items in FIFO order. Behavior `model_step` is provenance rather than a stale
admission gate, and producer envelopes, GAE segments, and Learner batches remain
distinct units. `action_mask` is stored and delivered unchanged like every other
sample field. Sample Pool does not interpret task action semantics or require masks
to be enabled.

Each `SamplePoolItem` retains the complete `behavior_model` and `producer` from
its accepted envelope, including across lease delivery, NACK and expiry. READY /
resident byte accounting includes the serialized transition and both source
fields. The
Learner checks the actual model source; the Pool never reconstructs provenance
from a local model, step or latest publication. Under its lock, `GetBatch` checks
cancellation and the request deadline before handling READY samples and again
after drawing, before creating a lease. An aborted draw restores the original
READY order and counters; committed leases retain the existing ACK/NACK/expiry
contract.

```bash
# When build/maze_sample_pool already exists
bash ./run.sh configs/pool_config.yaml
```

During full training the Learner container supervises this process; no separate launch is required.

## License

[MIT License](LICENSE)

## Build input and cache ownership

CMake/Ninja trees persist under `build/<platform>/<builder-image-id>/<channel>/` and ccache
uses a persistent Docker volume. Successful builds select `build/maze_sample_pool` for `run.sh`
and export the same binary. Run it in the same Linux toolchain environment; macOS hosts
only orchestrate Docker compilation. Toolchain identity partitions caches, not runtime admission.

Linux container builds refresh input attributes with `stat --cached=never` before CMake/Ninja, so a host edit is not hidden by virtiofs metadata caching. This neither changes source timestamps nor synchronizes upstream artifacts.
