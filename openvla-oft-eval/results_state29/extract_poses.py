import numpy as np
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

def obj_pose(env, obj):
    sim = env.env.sim
    bid = sim.model.body_name2id(obj.root_body)
    return np.array(sim.data.xpos[bid]), np.array(sim.data.xquat[bid])

def eef_pos(env):
    try:
        return np.array(env.env.robots[0].robot_model.eef_pos)
    except Exception:
        sim = env.env.sim
        for name in ['grip_site', 'right_grip_site', 'eef_site']:
            try:
                return np.array(sim.data.site_xpos[sim.model.site_name2id(name)])
            except Exception:
                continue
        return np.array([0, 0, 0])

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(4)
init_states = task_suite.get_task_init_states(4)

env, task_desc = get_libero_env(task, "openvla", resolution=256)

for sid in [28, 29, 30]:
    env.reset()
    obs = env.set_init_state(init_states[sid])
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(7))
    print(f"\n=== state {sid} ===")
    for obj in env.env.objects:
        pos, quat = obj_pose(env, obj)
        print(f"  obj={obj.name:24s} pos={np.round(pos,3)} quat={np.round(quat,3)}")
    print(f"  eef={np.round(eef_pos(env),3)}")
