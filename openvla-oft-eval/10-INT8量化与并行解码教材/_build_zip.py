#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把运维脚本打包成 zip。zip 内文件名全 ASCII，内容保持 LF 换行。"""
import os
import zipfile

SRC = r"C:\Users\27013\Desktop\oft\10-INT8量化与并行解码教材"
OUT = r"C:\Users\27013\Desktop\oft_upload\oft_ops_scripts.zip"

FILES = [
    "run_detached.sh",
    "diag_oom.sh",
    "_probe.sh",
    "memwatch.sh",
    "setup_swap.sh",
    "fix_crlf.sh",
    "resume_verify.sh",
    "download_oft_ckpt.sh",
    "dl_ckpt.py",
    "fix_activate_gl.sh",
    "fix_oft_gl_and_numpy.sh",
    "fix_oft_wandb.sh",
    "patch_oft_wandb_optional.sh",
    "setup_oft_env.sh",
]

README = """# OFT 运维脚本包 —— 使用说明

## 为什么必须「上传文件」而不是「粘贴进终端」

粘贴多行脚本进交互终端，脚本里的 `exit` 会直接关闭你的终端。
本包中部分脚本在参数不足时会 `exit 2`（这是正确行为），
但一旦被粘贴执行，退出的就是**你的交互 shell**，表现为
「终端异常退出，退出码 2」。

**退出码速查**（别把语法错当成 OOM）：

| 码  | 含义 |
|-----|------|
| 2   | bash 语法/解析失败，或脚本显式 exit 2（参数用错） |
| 1   | 脚本通用失败 |
| 126 | 不可执行（权限/格式） |
| 127 | 命令未找到 |
| 137 | 被 SIGKILL（128+9）—— 这才是被 OOM / nohang 杀 |

## 上传方法（任选其一）

1. DSW 控制台 → 文件浏览器 → 上传（把 zip 传到 /mnt/workspace/）
2. JupyterLab（实例常驻，端口 8088）→ 上传按钮
3. 命令行（小文件可用，本包不到 100KB）：
   scp oft_ops_scripts.zip root@<实例地址>:/mnt/workspace/

## 解压

    cd /mnt/workspace
    unzip -o oft_ops_scripts.zip
    chmod +x *.sh

## 先做这两步

    # 1) 清掉可能的 Windows 换行符（CRLF 会让 bash 报 syntax error 并退 2）
    bash fix_crlf.sh --check
    bash fix_crlf.sh

    # 2) 跑探针，结果写 /mnt/workspace/_probe.log
    bash _probe.sh

## 之后的用法

    # 脱离终端运行任何耗时命令（终端死了也不影响）
    bash run_detached.sh diag bash diag_oom.sh
    bash run_detached.sh ckpt bash experiments/robot/libero/failure_study/verify_oft_step.sh checkpoint

    # 看结果
    cat /mnt/workspace/_run_diag.latest.log

## 脚本清单

| 文件 | 用途 |
|------|------|
| _probe.sh          | 极简探针：磁盘/内存/cgroup/nohang/GPU，写 _probe.log |
| fix_crlf.sh        | 检测并清理 CRLF 与 BOM（`--check` 只报告） |
| run_detached.sh    | setsid+nohup 脱离终端运行命令，日志落盘 |
| diag_oom.sh        | 内存/磁盘/nohang 全量诊断 |
| memwatch.sh        | 内存哨兵：持续记录空闲内存水位，越线时抓 top RSS |
| setup_swap.sh      | 补 swap（自动避开 overlayfs 陷阱，小文件试挂后再放大） |
| resume_verify.sh   | 实例重启后七步复验（含自动重写 ~/.libero/config.yaml） |
| download_oft_ckpt.sh | checkpoint 探测/下载/校验（内置 unset MODELSCOPE_ENDPOINT） |
| dl_ckpt.py         | 纯 snapshot_download API 下载器（CLI 缺失时保底） |
| fix_activate_gl.sh | 把 activate_oft.sh 的 MUJOCO_GL 从 osmesa 改成 egl |
| setup_oft_env.sh   | 幂等补依赖 + 自检（环境补齐的唯一入口） |
| fix_oft_gl_and_numpy.sh | 清 pip 半成品残留 + 复验 numpy |
| fix_oft_wandb.sh   | 手工补 wandb 依赖（全 --no-deps） |
| patch_oft_wandb_optional.sh | 把仓库里 wandb 的裸 import 批量改成可选 |

## 本实例已知事实（2026-09-14 实测）

- 磁盘：`/` 492G，可用 **392G**（充裕）；`/tmp` 挂在 `/dev/vda3` 真实块设备
- 内存：28 GiB，**Swap 0B**（硬伤）
- **cgroup 从未发生过 OOM**：`memory.failcnt=0`、
  `memory.oom_control: oom_kill 0, under_oom 0`、历史峰值仅 **1.33 GiB**
- `nohang` 守护进程常驻（PID 208/236/438），配置文件
  `/etc/dsw/cache/pre-install/plugins/nohang/v0.1.0/pai-nohang.conf`：
  - `soft_threshold_min_mem = 3%`（≈860 MiB 空闲）→ 软动作
  - `hard_threshold_min_mem = 2%`（≈573 MiB 空闲）→ 硬动作
  - `kill_process_group_enabled = True` → **硬动作杀整个进程组**
  - `psi_checking_enabled = True` 但 `/proc/pressure/memory` 不存在 → PSI 判据失效
- 容器**未暴露** `/sys/fs/cgroup/memory.events` 与 PSI
- GPU：NVIDIA A10，23028 MiB，当前 0 MiB 占用
- bash 5.1.16

## 加 swap 的正确姿势

`/` 是 **overlayfs**，`swapon` 在其上通常失败。
`setup_swap.sh` 会自动探测非 overlay 的可写目录（`/tmp` 可用），
先用 64 MiB 小文件试挂验证可行性，成功后才放大到 32G。

    bash setup_swap.sh --check    # 只探测可行性
    bash setup_swap.sh            # 实际创建（默认 32G）
    SWAP_GB=16 bash setup_swap.sh # 指定大小
"""


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)

    missing = []
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in FILES:
            p = os.path.join(SRC, name)
            if not os.path.exists(p):
                missing.append(name)
                continue
            data = open(p, "rb").read()
            # 强制 LF，去 BOM
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            if data[:3] == b"\xef\xbb\xbf":
                data = data[3:]
            zi = zipfile.ZipInfo(name, date_time=time_tuple())
            zi.external_attr = 0o755 << 16
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, data)

        zi = zipfile.ZipInfo("README_上传说明.md", date_time=time_tuple())
        z.writestr(zi, README.encode("utf-8"))

    size = os.path.getsize(OUT)
    print(f"written: {OUT}")
    print(f"size   : {size:,} bytes ({size/1024:.1f} KiB)")
    with zipfile.ZipFile(OUT) as z:
        names = z.namelist()
        print(f"files  : {len(names)}")
        for n in names:
            print(f"   {z.getinfo(n).file_size:>8,}  {n}")
    if missing:
        print("MISSING (未找到，未打包):", missing)


def time_tuple():
    import time
    return time.localtime()[:6]


if __name__ == "__main__":
    main()
