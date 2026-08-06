# RL Sample Pool

简体中文 | [English](README.en.md)

C++ 本地样本服务制品仓库，当前版本为 `0.8.1`。`maze_sample_distributor` 在本地训练中合并提供样本接入与单 consumer 内存租约池，由 Learner 镜像监管，不单独启动任务容器。它不是 Reverb 的等价实现。

## 构建

先构建 `rl-contracts`，再执行：

```bash
bash build_artifact.sh
```

输出目录：

```text
../.workspace/artifacts/rl-sample-pool/0.8.1/<platform>/
```

首次生成一个版本的制品时，仓库必须位于已经审核并提交的 clean Git 保存点；构建
入口拒绝从 dirty worktree 创建新版本。已存在的同版本制品只允许在源码身份和文件
checksum 全部一致时复用。

将对应平台的制品显式复制到 `rl-learner/sample-pool/` 后，再构建 Learner 镜像。

## License

[MIT License](LICENSE)
