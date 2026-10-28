# Interview Workspace

三个项目的当前源码、文档、模型、样例与实验资源汇总。

| 项目 | 内容 | 入口 |
| --- | --- | --- |
| digital-human-sdk | C++ 数字人 SDK、音视频流水线、推理与渲染优化 | [项目说明](digital-human-sdk/README.md) |
| moguwulian-agent-backend | Agent 后端、节能测算、报价流程与项目资料 | [项目说明](moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/README.md) |
| openvla-oft-eval | OpenVLA-OFT 评测、量化、微调与机器人实验 | [项目说明](openvla-oft-eval/README.md) |

## 获取完整资源

模型、数据集及较大的资源文件使用 Git LFS。安装 Git LFS 后运行：

```bash
git lfs install
git clone https://github.com/lijiaming-boop/Interview-Workspace.git
cd Interview-Workspace
git lfs pull
```

本仓库保存工作区当前文件快照，各项目原有 Git 历史不在此仓库中合并。依赖环境、缓存、临时运行文件、本地数据库和真实密钥不上传；配置示例保留。构建与运行方法见各项目文档。

## 许可

各项目与上游依赖的许可证、版权及模型使用条款以其各自文件为准；本汇总仓库不变更这些条款。
