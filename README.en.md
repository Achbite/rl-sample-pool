# RL Sample Pool

English | [简体中文](README.md)

C++ local sample-service artifact repository, currently at version `0.8.0`. For local training, `maze_sample_distributor` combines sample ingress with a single-consumer in-memory lease pool. It is supervised by the Learner image, is not a separate task container, and is not equivalent to Reverb.

## Build

Build `rl-contracts` first, then run:

```bash
bash build_artifact.sh
```

Output directory:

```text
../.workspace/artifacts/rl-sample-pool/0.8.0/<platform>/
```

Copy the artifact for the selected platform into `rl-learner/sample-pool/` before building the Learner image.

## License

[MIT License](LICENSE)
