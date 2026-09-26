# RL Sample Pool

简体中文 | [English](README.en.md)

## 1. 运行测试

在已经挂载 Contracts C++ bindings 的构建容器中执行：

```bash
bash ./test.sh
```

`test.sh` 是本仓库的统一测试入口，并按显式清单运行当前开发校验。

## 2. 构建当前制品

显式提供已经生成的 Contracts Training 制品目录（其中包含 `cpp/` 与 `sdk/`）：

```bash
CONTRACT_CPP_DIR=/path/to/training/cpp bash ./build.sh
```

本仓只构建自身，不构建或同步上游。构建脚本只编译正式二进制；测试仍通过 `bash ./test.sh` 显式运行。
默认输出到 `build/artifacts/production/<platform>/`；调用者可以用 `RL_ARTIFACT_ROOT` 指定输出根目录。
`bash build_dev_artifact.sh` 使用相同实现，默认输出到 `build/artifacts/development/<platform>/current/`。

## 3. 装配到 Learner

在遵循 Framework 同级仓库布局的工作区，从 `rl-framework` 执行：

```bash
bash ./sync_artifacts.sh
# 开发制品通道
bash ./sync_artifacts.sh --development
```

Framework 显式调用 Contracts、Pool 和 Distributor 各自的构建入口，再将所选制品装配到 Learner。
组件仓不依赖这种目录布局，单独构建也不会修改 Learner。

## 4. 单独运行

当前 backend 是非破坏性 Replay Pool：样本写入后持续驻留，采样只生成独立 batch 快照。
`sampling.strategy` 默认为 `uniform_without_replacement`，仅禁止同一个 batch 内重复；
可改为 `uniform_with_replacement`，允许同一 item 在 batch 内出现多次。两种模式都允许跨 batch
复用，且都保留“驻留样本数达到 requested_transitions 后才返回完整 batch”的就绪门槛。

ACK、NACK 和租约到期只结算 batch，不删除、恢复或刷新池内样本。容量压力仍按原始插入顺序
FIFO 淘汰最老驻留对象，包含正在被某个 batch 使用的原对象；已经返回的快照不受淘汰影响。
`ready_transitions` 等于可采样的驻留数量，`leased_transitions` 则是独立在途 batch 的位置数，
不能把两者相加当作 resident。仍只允许一个在途 batch，不扩展并行 Learner。

每个 `SamplePoolItem` 保留 Envelope 的完整 `behavior_model`、`producer`、原 item ID 与插入事实。
`draw_count` 记录实际累计抽样位置次数；同 batch 重复位置携带相同的批次末快照。
驻留字节包含 transition 和两个来源字段，不包含已经复制给消费者的 batch。
Pool 不解释任务、动作 mask、模型兼容性或陈旧度，不重算 GAE、不生成替代身份。

`GetBatch` 在锁内于选择前和提交 draw_count/交付租约前检查取消和期限；未提交的取消抽样不改变
驻留顺序或计数。TRAINED/INVALID/SHUTDOWN_UNTRAINED 只统计明确 ACK 的 batch 位置；
`acknowledged_transition_slot_count` 是所有已 ACK 位置，`released_transition_slot_count` 是
NACK/到期释放的位置。finalize 清理驻留数据，单独计入 `finalized_transition_count`，不将 Replay
尾部误记为未训练。两套计数分别满足：

```text
accepted_unique = resident + evicted + finalized
drawn_slots = acknowledged_slots + released_slots + leased_slots
acknowledged_slots = trained + invalid + shutdown_untrained
```

```bash
# 已有本地 build/maze_sample_pool 时
bash ./run.sh configs/pool_config.yaml
```

完整训练由 Learner 容器监管该进程，不需要单独启动。

## License

[MIT License](LICENSE)

## 构建与运行路径

CMake/Ninja 构建目录位于 `build/<platform>/<builder-image-id>/<channel>/`，ccache 使用持久 Docker 卷。构建成功后 `build/maze_sample_pool` 指向本次二进制，同时导出制品；在相同 Linux 工具链环境中执行 `bash run.sh` 使用该输出。macOS 宿主负责 Docker 构建，不直接执行 Linux ELF。正常重建不清理缓存；构建镜像身份仅隔离工具链缓存，不参与运行准入。

Linux 容器构建前用 `stat --cached=never` 刷新实际输入的文件属性，避免 virtiofs 暂存旧 mtime 导致刚保存的宿主源码被误判为 no-op；不会修改源码时间戳或同步上游制品。
