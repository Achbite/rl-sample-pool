# RL Sample Pool

简体中文 | [English](README.en.md)

## 1. 运行测试

在已经挂载 Contracts C++ bindings 的构建容器中执行：

```bash
bash ./test.sh
```

`test.sh` 是本仓库的统一测试入口，并按显式清单运行当前开发校验。

## 2. 构建 0.14.0 制品

先生成 Contracts 制品，再从 clean Git 保存点执行：

```bash
(cd ../rl-contracts && bash build_artifact.sh)
bash build_artifact.sh
```

构建脚本会编译并运行测试。输出：

```text
../.workspace/artifacts/rl-sample-pool/0.14.0/<platform>/
```

## 3. 装配到 Learner

不要手工复制二进制。使用 Learner 同步入口：

```bash
(cd ../rl-learner && bash scripts/sync_runtime_artifacts.sh)
```

## 4. 单独运行

当前 backend 存储独立 processed transition。训练 draw 在 READY 集合中执行 Uniform
无放回选择；TRAINED ACK 后删除，NACK 恢复 READY，容量压力只按 FIFO 淘汰最老 READY Item。
它不按 behavior `model_step` 做陈旧度门禁，也不把 Producer envelope、GAE segment 或 Learner
train batch 混成同一个容量单位。

```bash
# 已有本地 build/maze_sample_pool 时
bash ./run.sh configs/pool_config.yaml
```

完整训练由 Learner 容器监管该进程，不需要单独启动。

## License

[MIT License](LICENSE)
