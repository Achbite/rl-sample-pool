# RL Sample Pool

简体中文 | [English](README.en.md)

C++ 本地样本服务制品仓库，当前版本为 `0.6.0`。`maze_sample_distributor` 在本地训练中合并提供样本接入与单 consumer 内存租约池，由 Learner 镜像监管，不单独启动任务容器。它不是 Reverb 的等价实现。

## 构建

先构建 `rl-contracts`，再执行：

```bash
bash build_artifact.sh
```

输出目录：

```text
../.workspace/artifacts/rl-sample-pool/0.6.0/<platform>/
```

将对应平台的制品显式复制到 `rl-learner/sample-pool/` 后，再构建 Learner 镜像。

## License

[MIT License](LICENSE)
