"""Extract expert correction windows for the known failing initial states."""

import argparse
from pathlib import Path

import h5py


def copy_node(source, target, start, end, trajectory_length):
    for key, value in source.attrs.items():
        target.attrs[key] = value
    for name, child in source.items():
        if isinstance(child, h5py.Group):
            group = target.create_group(name)
            copy_node(child, group, start, end, trajectory_length)
        else:
            array = child[start:end] if child.ndim and child.shape[0] == trajectory_length else child[()]
            target.create_dataset(name, data=array)
            for attr, value in child.attrs.items():
                target[name].attrs[attr] = value


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--demos", nargs="+", required=True, type=int)
parser.add_argument("--start", type=int, default=45)
parser.add_argument("--end", type=int, default=91)
args = parser.parse_args()

if args.end <= args.start:
    raise SystemExit("end must be greater than start")
args.output.parent.mkdir(parents=True, exist_ok=True)

with h5py.File(args.input, "r") as source, h5py.File(args.output, "w") as target:
    for key, value in source.attrs.items():
        target.attrs[key] = value
    out_data = target.create_group("data")
    for demo_id in args.demos:
        name = f"demo_{demo_id}"
        if name not in source["data"]:
            raise KeyError(f"missing {name}")
        src_demo = source["data"][name]
        n = int(src_demo["actions"].shape[0])
        if not 0 <= args.start < args.end <= n:
            raise ValueError(f"{name}: window [{args.start}, {args.end}) outside length {n}")
        dst_demo = out_data.create_group(name)
        copy_node(src_demo, dst_demo, args.start, args.end, n)
        dst_demo["dones"][...] = 0
        dst_demo["dones"][-1] = 1
        print(f"{name}: source_length={n}, correction_window=[{args.start}, {args.end}), output_length={len(dst_demo['dones'])}")

print(f"saved: {args.output}")
