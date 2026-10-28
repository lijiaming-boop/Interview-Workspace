# 深度学习人脸模型

`FaceDetector` 不再依赖 dlib。它使用以下模型组合：

| 职责 | 模型 | 运行时 | 原因 |
| --- | --- | --- | --- |
| 人脸检测 | SCRFD-2.5G-KPS | ncnn | 对小脸、侧脸比 HOG 稳定；模型小，且项目已链接 ncnn。 |
| 稠密关键点 | 2D106 | OpenCV DNN (ONNX) | 保留现有 68 点对齐与口部遮罩的输出契约；106 点会在 SDK 内映射为 iBUG 68 点。 |

将模型放在同一个目录（例如 `models/face/`）：

```text
models/face/
  scrfd_2.5g_kps-opt2.param
  scrfd_2.5g_kps-opt2.bin
  2d106det.onnx
```

然后将该**目录**传给 `FaceDetector::loadModel` 或
`Pipeline::SetLandmarkModelPath`（为保持源代码兼容性，后者沿用了旧名称）。

SCRFD 的 ncnn 权重必须是带 5 点输出、输出 blob 名为
`score_8/16/32` 与 `bbox_8/16/32` 的 `*-opt2` 版本。当前已验证可用的 GitHub
来源是 [xlite-dev/scrfd-toolkit](https://github.com/xlite-dev/scrfd-toolkit) 中的
`scrfd_2.5g_bnkps_shape640x640.opt.param/.bin`，接入时改名为上面的项目文件名。
该导出把输出展平为 `[1,N,1]` / `[1,N,4]`，两个 anchor 按每个网格点交错排列；SDK
按这一布局解析，并使用 640x640 letterbox、RGB、均值 127.5、归一化 1/128 的预处理。
2D106 模型的输入为 RGB `192x192`，输出为 212 个归一化坐标；当前文件来自
InsightFace 官方 GitHub `buffalo_m` 模型包。

模型文件被 `.gitignore` 排除，避免把大体积权重和其许可义务混入 SDK 源码。上线前必须单独
审查所选权重及训练数据的许可证；InsightFace 发布的部分预训练模型仅限非商业研究，商业 SDK
应使用已获授权或自行训练并导出的同结构模型。
