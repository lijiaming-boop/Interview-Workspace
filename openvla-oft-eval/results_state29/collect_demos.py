"""M3: 脚本策略在 state 29 及邻域(28~32)采集专家示教（OSC 归一化动作版）"""
import numpy as np, h5py
from transforms3d.euler import quat2euler
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

STATES = [28, 29, 30, 31, 32]
DEMOS_PER_STATE = 2
OUT = "/mnt/workspace/demos_state29.hdf5"
EEF_BODY = "gripper0_eef"
ACT_SCALE = 0.05   # OSC: 输入[-1,1] -> 每步 ±0.05m

def obj_pos(env, name):
    sim = env.env.sim
    for cand in (name + "_main", name):
        try:
            return np.array(sim.data.xpos[sim.model.body_name2id(cand)])
        except ValueError:
            continue
    raise ValueError(f"body {name} not found")

def eef_pos(env):
    return np.array(env.env.sim.data.xpos[env.env.sim.model.body_name2id(EEF_BODY)])

def eef_state(env):
    sim = env.env.sim
    pos = np.array(sim.data.xpos[sim.model.body_name2id(EEF_BODY)])
    quat = np.array(sim.data.xquat[sim.model.body_name2id(EEF_BODY)])
    return np.concatenate([pos, np.array(quat2euler(quat))])

def gripper_state(env):
    sim = env.env.sim
    robot = env.env.robots[0]
    idx = [sim.model.joint_name2id(j) for j in robot.gripper.joints]
    return np.array([sim.data.qpos[i] for i in idx])

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(4)
init_states = task_suite.get_task_init_states(4)
env, task_desc = get_libero_env(task, "openvla", resolution=256)

KEYS = {}
rec = None
def step(action):
    obs, r, done, _ = env.step(action)
    if not KEYS:
        KEYS["third"] = next(k for k in ["agentview_image", "agentview_rgb"] if k in obs)
        KEYS["wrist"] = next(k for k in ["robot0_eye_in_hand_image", "robot0_eye_in_hand_rgb"] if k in obs)
    rec["image"].append(obs[KEYS["third"]][::-1])
    rec["wrist_image"].append(obs[KEYS["wrist"]][::-1])
    rec["eef_state"].append(eef_state(env))
    rec["gripper_state"].append(gripper_state(env))
    rec["action"].append(action)

def move_to(target, n_steps, grip):
    for _ in range(n_steps):
        delta = np.clip(target - eef_pos(env), -0.05, 0.05)
        cmd = np.clip(delta / ACT_SCALE, -1, 1)          # 归一化到 OSC 输入范围
        step(np.concatenate([cmd, [0, 0, 0], [grip]]))

f = h5py.File(OUT, "w")
data_grp = f.create_group("data")
n_saved = 0

for sid in STATES:
    for trial in range(DEMOS_PER_STATE):
        env.reset()
        obs = env.set_init_state(init_states[sid])
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(7))

        bowl_p = obj_pos(env, "akita_black_bowl_1")
        plate_p = obj_pos(env, "plate_1")
        rec = {"image": [], "wrist_image": [], "eef_state": [], "gripper_state": [], "action": []}

        move_to(bowl_p + [0, 0, 0.15], 40, 1)     # 上方接近
        move_to(bowl_p + [0, 0, 0.03], 30, 1)     # 下降
        for _ in range(15):                       # 抓取
            step(np.concatenate([np.zeros(3), [0, 0, 0], [-1]]))
        move_to(bowl_p + [0, 0, 0.25], 40, -1)    # 抬升
        move_to(plate_p + [0, 0, 0.15], 60, -1)   # 搬运
        move_to(plate_p + [0, 0, 0.06], 30, -1)   # 下降
        for _ in range(15):                       # 释放
            step(np.concatenate([np.zeros(3), [0, 0, 0], [1]]))
        move_to(plate_p + [0, 0, 0.20], 20, 1)    # 撤离

        bp, pp = obj_pos(env, "akita_black_bowl_1"), obj_pos(env, "plate_1")
        ok = (np.linalg.norm(bp[:2] - pp[:2]) < 0.06) and (pp[2] < bp[2] < pp[2] + 0.15)
        print(f"state {sid} trial {trial}: success={ok} bowl={np.round(bp,3)}")
        if not ok:
            continue
        g = data_grp.create_group(f"demo_{n_saved}")
        g.attrs["init_state_id"] = sid
        for k, v in rec.items():
            g.create_dataset(k, data=np.array(v))
        n_saved += 1

f.attrs["num_demos"] = n_saved
f.close()
print(f"saved {n_saved} demos -> {OUT}")
