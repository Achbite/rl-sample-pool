# RL Sample Pool

简体中文 | [English](README.en.md)

## 1. 运行测试

在已经挂载 Contracts C++ bindings 的构建容器中执行：

```bash
bash ./test.sh
```

`test.sh` 是本仓库的统一测试入口，并按显式清单运行当前开发校验。

## 2. 构建当前制品

先生成任务无关的训练协议制品，再从当前源码执行：

```bash
(cd ../rl-contracts && bash build_artifact.sh training)
bash build_artifact.sh
```

构建脚本只编译正式二进制；测试仍只通过第 1 节的 `bash ./test.sh` 显式运行。输出：

```text
../.workspace/artifacts/rl-sample-pool/<platform>/
```

## 3. 装配到 Learner

推荐使用 Learner 同步入口复制完整制品：

```bash
(cd ../rl-learner && bash scripts/sync_runtime_artifacts.sh)
```

开发制品可由 Learner 的 `make deps` 统一构建和同步；`make shell` 不会隐式执行该操作。

## 4. 单独运行

当前 backend 存储独立 processed transition。训练 draw 在 READY 集合中执行 Uniform
无放回选择；TRAINED ACK 后删除，NACK 恢复 READY，容量压力只按 FIFO 淘汰最老 READY Item。
它不按 behavior `model_step` 做陈旧度门禁，也不把 Producer envelope、GAE segment 或 Learner
train batch 混成同一个容量单位。`action_mask` 与其他样本字段一样按字节合同原样保存和交付；
Sample Pool 不判断任务动作语义，也不要求 mask 必须启用。

每个 `SamplePoolItem` 保留其输入 Envelope 的完整 `behavior_model` 和 `producer`。
READY / resident 的字节计量包含 transition 及这两个来源字段的序列化大小。
租约投递、NACK 和到期回队列均携带相同来源；Learner 根据实际来源判断模型身份，
Pool 不根据本地模型、step 或最新发布记录补写来源。`GetBatch` 在锁内检查取消和请求
期限，再处理 READY 样本；抽样后、建立租约前再次检查。未提交的取消抽样恢复原始 READY
顺序和计数，已建立的租约继续按现有 ACK/NACK/到期合同处理。

```bash
# 已有本地 build/maze_sample_pool 时
bash ./run.sh configs/pool_config.yaml
```

完整训练由 Learner 容器监管该进程，不需要单独启动。

## License

[MIT License](LICENSE)
