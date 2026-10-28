"""
video_chunk_analysis.py
从失败视频提取帧，做 chunk 边界时间对齐分析。
用法：python video_chunk_analysis.py <视频目录路径>
"""
import glob, sys
import cv2
import numpy as np

WAIT_STEPS = 10     # num_steps_wait
VIDEO_FPS = 30.0    # 回放帧率
CTRL_HZ = 20.0      # 控制频率


def video_sec_to_step(vs):
    """视频秒 -> 仿真步"""
    return WAIT_STEPS + round(vs * CTRL_HZ)


def step_to_video_sec(t):
    """仿真步 -> 视频秒"""
    return (t - WAIT_STEPS) / VIDEO_FPS


def detect_gripper_open_ratio(frame):
    """
    估计夹爪开合度：取画面上部区域（机械臂/夹爪位置），检测深色（夹爪手指）像素的水平分布宽度。
    夹爪闭合时深色区域窄，张开时宽。
    返回 0~1 之间的值（归一化宽度）。
    """
    h, w = frame.shape[:2]
    # 取画面上部 1/3 区域（机械臂和夹爪通常在这里）
    roi = frame[:h // 3, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 检测深色像素（夹爪手指是深灰色/黑色）
    dark_mask = gray < 80
    # 找每行深色像素的左右边界
    rows_with_dark = np.where(dark_mask.any(axis=1))[0]
    if len(rows_with_dark) < 3:
        return 0.0
    # 取中间 50% 的行（避开边缘噪声）
    mid_start = len(rows_with_dark) // 4
    mid_end = 3 * len(rows_with_dark) // 4
    selected_rows = rows_with_dark[mid_start:mid_end]
    if len(selected_rows) == 0:
        return 0.0
    # 计算这些行中深色像素的水平跨度
    widths = []
    for r in selected_rows:
        cols = np.where(dark_mask[r])[0]
        if len(cols) > 0:
            widths.append(cols.max() - cols.min())
    if not widths:
        return 0.0
    avg_width = np.mean(widths)
    # 归一化到 0~1（假设最大宽度约为 ROI 宽度的 60%）
    return float(avg_width / (w * 0.6))


def analyze_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    fname = path.replace("\\", "/").split("/")[-1]
    print(f"\n{'=' * 70}")
    print(f"文件: {fname}")
    print(f"帧率={fps:.1f}, 总帧数={total_frames}, 时长={duration:.1f}s")
    print(f"{'=' * 70}")

    # 逐帧读取
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"实际读取 {len(frames)} 帧")

    # 计算每帧的 gripper 开合度
    gripper_ratios = []
    for f in frames:
        gripper_ratios.append(detect_gripper_open_ratio(f))
    gripper_ratios = np.array(gripper_ratios)

    # 逐帧对应仿真步
    # 仿真步 t = 10 + round(frame_idx / fps * ctrl_hz)
    steps = np.array([WAIT_STEPS + round(i / fps * CTRL_HZ) for i in range(len(frames))])

    # 输出 chunk 边界时间映射
    print("\n--- chunk 边界时间映射（LIBERO: chunk=8步, 20Hz控制, 30fps回放）---")
    print("chunk边界t = 10 + 8k (k=0,1,2,...)")
    print("对应视频秒 = (t-10)/30")
    max_t = steps[-1] if len(steps) else 0
    k = 0
    while True:
        t = WAIT_STEPS + 8 * k
        vs = step_to_video_sec(t)
        if vs > duration + 0.5:
            break
        frame_idx = int(round(vs * fps))
        if frame_idx < len(gripper_ratios):
            gr = gripper_ratios[frame_idx]
            print(f"  chunk {k:2d}: t={t:3d} (视频{vs:.2f}s, 帧{frame_idx}) | gripper_ratio={gr:.3f}")
        else:
            print(f"  chunk {k:2d}: t={t:3d} (视频{vs:.2f}s) | 超出帧范围")
        k += 1

    # 检测 gripper 开合变化点
    print("\n--- gripper 开合变化检测（chunk 内）---")
    # 平滑
    if len(gripper_ratios) > 5:
        smoothed = np.convolve(gripper_ratios, np.ones(5) / 5, mode='same')
    else:
        smoothed = gripper_ratios

    # 找局部极值（开合切换）
    transitions = []
    for i in range(1, len(smoothed)):
        if smoothed[i - 1] > 0.02 and smoothed[i] < 0.01:
            transitions.append(('close', i, steps[i], step_to_video_sec(steps[i])))
        elif smoothed[i - 1] < 0.01 and smoothed[i] > 0.02:
            transitions.append(('open', i, steps[i], step_to_video_sec(steps[i])))

    for kind, fi, t, vs in transitions:
        # 判断是否在 chunk 边界
        chunk_idx = (t - WAIT_STEPS) // 8
        inner_idx = (t - WAIT_STEPS) % 8
        where = "chunk 边界 (index 0)" if inner_idx == 0 else f"chunk {chunk_idx} 内 index {inner_idx}"
        print(f"  [{kind}] 帧{fi} t={t} (视频{vs:.2f}s) => {where}")

    if not transitions:
        print("  未检测到明显 gripper 开合变化（阈值可能需要调整）")

    # 统计
    print(f"\n--- 统计 ---")
    print(f"  平均 gripper_ratio: {gripper_ratios.mean():.4f}")
    print(f"  最大: {gripper_ratios.max():.4f} (帧{gripper_ratios.argmax()})")
    print(f"  最小: {gripper_ratios.min():.4f} (帧{gripper_ratios.argmin()})")

    return gripper_ratios, steps


if __name__ == "__main__":
    default_dir = "C:\\Users\\27013\\Desktop\\oft\\04-评测结果\\failure"
    paths = sys.argv[1:] or glob.glob(default_dir + "\\*.mp4")
    if not paths:
        print("未找到视频文件")
        sys.exit(1)
    for p in sorted(paths):
        analyze_video(p)
