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

## 构建与运行路径

CMake/Ninja 构建目录位于 `build/<platform>/<builder-image-id>/<channel>/`，ccache 使用持久 Docker 卷。构建成功后 `build/maze_sample_pool` 指向本次二进制，同时导出制品；在相同 Linux 工具链环境中执行 `bash run.sh` 使用该输出。macOS 宿主负责 Docker 构建，不直接执行 Linux ELF。正常重建不清理缓存；构建镜像身份仅隔离工具链缓存，不参与运行准入。

Linux 容器构建前用 `stat --cached=never` 刷新实际输入的文件属性，避免 virtiofs 暂存旧 mtime 导致刚保存的宿主源码被误判为 no-op；不会修改源码时间戳或同步上游制品。
