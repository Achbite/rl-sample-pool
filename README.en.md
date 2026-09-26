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

The backend is a non-destructive Replay Pool: inserted samples remain resident,
while draws return independent batch snapshots. `sampling.strategy` defaults to
`uniform_without_replacement`, preventing duplicate item IDs only within a batch.
`uniform_with_replacement` allows repeated slots. Both modes allow reuse across
batches and retain the existing readiness threshold of at least
`requested_transitions` resident items before delivering a complete batch.

ACK, NACK and lease expiry settle only the batch. They never remove, reinsert or
refresh resident items. FIFO capacity eviction follows original insertion order,
including items used by an in-flight batch; delivered snapshots remain unchanged.
`ready_transitions` is the eligible resident count, while `leased_transitions`
counts independent in-flight batch slots. They are not disjoint storage partitions.
The existing single in-flight batch limit remains; this adds no parallel Learners.

Each `SamplePoolItem` retains its original item ID, complete behavior model,
producer and insertion facts. `draw_count` counts cumulative committed draw slots;
repeated slots in a batch share the same end-of-batch snapshot. Resident byte
accounting includes the transition and both source fields, not consumer snapshots.
The Pool does not interpret task, action-mask or model compatibility semantics,
apply staleness gates, recompute GAE or reconstruct provenance.

Under its lock, `GetBatch` checks cancellation/deadline before selection and before
committing draw history and the delivery lease. Cancelled selections leave resident
order and counters untouched. TRAINED/INVALID/SHUTDOWN_UNTRAINED count explicitly
acknowledged batch slots. `acknowledged_transition_slot_count` totals ACKed slots;
`released_transition_slot_count` counts NACK/expired slots. Finalization clears
residents into `finalized_transition_count`, not SHUTDOWN_UNTRAINED. Accounting is:

```text
accepted_unique = resident + evicted + finalized
drawn_slots = acknowledged_slots + released_slots + leased_slots
acknowledged_slots = trained + invalid + shutdown_untrained
```

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
