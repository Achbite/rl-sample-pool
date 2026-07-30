# RL Sample Pool

English | [简体中文](README.md)

C++ sample-pool artifact repository, currently at version `0.5.0`. The generated SampleDistributor binary is used by the AIServer training image and is not started as a separate task container.

## Build

Build `rl-contracts` first, then run:

```bash
bash build_artifact.sh
```

Output directory:

```text
../.workspace/artifacts/rl-sample-pool/0.5.0/<platform>/
```

Copy the artifact for the selected platform into `rl-aiserver/sample-distributor/` before building the AIServer image.

## License

[MIT License](LICENSE)
