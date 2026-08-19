# RL Sample Pool

简体中文 | [English](README.en.md)

## 1. 运行测试

在已经挂载 Contracts C++ bindings 的构建容器中执行：

```bash
bash ./test.sh
```

`test.sh` 是本仓库唯一公开测试入口；新增或扩写测试必须先有用户批准的 TCR。

## 2. 构建 0.13.0 制品

先生成 Contracts 制品，再从 clean Git 保存点执行：

```bash
(cd ../rl-contracts && bash build_artifact.sh)
bash build_artifact.sh
```

构建脚本会编译并运行测试。输出：

```text
../.workspace/artifacts/rl-sample-pool/0.13.0/<platform>/
```

## 3. 装配到 Learner

不要手工复制二进制。使用 Learner 同步入口：

```bash
(cd ../rl-learner && bash scripts/sync_runtime_artifacts.sh)
```

## 4. 单独运行

```bash
./maze_sample_pool configs/pool_config.yaml
```

完整训练由 Learner 容器监管该进程，不需要单独启动。

## License

[MIT License](LICENSE)
