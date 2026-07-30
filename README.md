# RL Sample Pool

简体中文 | [English](README.en.md)

C++ 样本池制品仓库，当前版本为 `0.5.0`。生成的 SampleDistributor 二进制由 AIServer 训练镜像使用，不单独启动任务容器。

## 构建

先构建 `rl-contracts`，再执行：

```bash
bash build_artifact.sh
```

输出目录：

```text
../.workspace/artifacts/rl-sample-pool/0.5.0/<platform>/
```

将对应平台的制品显式复制到 `rl-aiserver/sample-distributor/` 后，再构建 AIServer 镜像。

## License

[MIT License](LICENSE)
