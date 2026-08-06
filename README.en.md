# RL Sample Pool

English | [简体中文](README.md)

C++ local sample-service artifact repository, currently at version `0.8.1`. For local training, `maze_sample_distributor` combines sample ingress with a single-consumer in-memory lease pool. It is supervised by the Learner image, is not a separate task container, and is not equivalent to Reverb.

## Build

Build `rl-contracts` first, then run:

```bash
bash build_artifact.sh
```

Output directory:

```text
../.workspace/artifacts/rl-sample-pool/0.8.1/<platform>/
```

Creating a version for the first time requires a reviewed and committed clean
Git savepoint. The build entrypoint refuses to create a new artifact from a
dirty worktree. An existing artifact can only be reused when both source
identity and every file checksum still match.

Copy the artifact for the selected platform into `rl-learner/sample-pool/` before building the Learner image.

## License

[MIT License](LICENSE)
