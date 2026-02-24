#!/usr/bin/env python3
"""
生成测试用相机轨迹文件 (JSON 格式).

支持的轨迹类型:
  - orbit: 绕原点做水平圆弧
  - zoom_in: 镜头推进 (向前移动)
  - zoom_out: 镜头拉远 (向后移动)
  - pan_left: 水平左平移
  - pan_right: 水平右平移
  - tilt_up: 镜头仰角
  - tilt_down: 镜头俯角
  - static: 静止 (identity)

用法:
  python generate_camera_poses.py --type orbit --num_frames 81 --output orbit.json
"""

import argparse
import json
import math
import numpy as np


def look_at(eye, target, up=np.array([0.0, 1.0, 0.0])):
    """Compute camera-to-world matrix given eye, target, up."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def generate_orbit(num_frames, radius=3.0, elevation=0.0, angle_range=60.0):
    """水平圆弧轨迹."""
    c2ws = []
    for i in range(num_frames):
        angle = math.radians(-angle_range / 2 + angle_range * i / (num_frames - 1))
        eye = np.array([radius * math.sin(angle), elevation, radius * math.cos(angle)])
        c2w = look_at(eye, np.zeros(3))
        c2ws.append(c2w.tolist())
    return c2ws


def generate_zoom(num_frames, start_z=5.0, end_z=2.0):
    """推/拉镜头."""
    c2ws = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        z = start_z + (end_z - start_z) * t
        eye = np.array([0.0, 0.0, z])
        c2w = look_at(eye, np.zeros(3))
        c2ws.append(c2w.tolist())
    return c2ws


def generate_pan(num_frames, direction="left", distance=2.0):
    """水平平移."""
    c2ws = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        if direction == "left":
            x = -distance * t
        else:
            x = distance * t
        eye = np.array([x, 0.0, 3.0])
        target = np.array([x, 0.0, 0.0])
        c2w = look_at(eye, target)
        c2ws.append(c2w.tolist())
    return c2ws


def generate_tilt(num_frames, direction="up", angle_range=30.0, radius=3.0):
    """仰角/俯角."""
    c2ws = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        if direction == "up":
            angle = math.radians(angle_range * t)
        else:
            angle = math.radians(-angle_range * t)
        eye = np.array([0.0, radius * math.sin(angle), radius * math.cos(angle)])
        c2w = look_at(eye, np.zeros(3))
        c2ws.append(c2w.tolist())
    return c2ws


def generate_static(num_frames):
    """静止相机."""
    c2w = look_at(np.array([0.0, 0.0, 3.0]), np.zeros(3))
    return [c2w.tolist()] * num_frames


def main():
    parser = argparse.ArgumentParser(description="Generate camera trajectory JSON files")
    parser.add_argument("--type", choices=["orbit", "zoom_in", "zoom_out", "pan_left", "pan_right", "tilt_up", "tilt_down", "static"], default="orbit")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--output", type=str, default="camera_poses.json")
    parser.add_argument("--output_dir", type=str, default=None, help="If set, generates all types into this directory")
    args = parser.parse_args()

    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        types = ["orbit", "zoom_in", "zoom_out", "pan_left", "pan_right", "tilt_up", "tilt_down", "static"]
        for t in types:
            output_path = os.path.join(args.output_dir, f"{t}.json")
            if t == "orbit":
                c2ws = generate_orbit(args.num_frames)
            elif t == "zoom_in":
                c2ws = generate_zoom(args.num_frames, 5.0, 2.0)
            elif t == "zoom_out":
                c2ws = generate_zoom(args.num_frames, 2.0, 5.0)
            elif t == "pan_left":
                c2ws = generate_pan(args.num_frames, "left")
            elif t == "pan_right":
                c2ws = generate_pan(args.num_frames, "right")
            elif t == "tilt_up":
                c2ws = generate_tilt(args.num_frames, "up")
            elif t == "tilt_down":
                c2ws = generate_tilt(args.num_frames, "down")
            elif t == "static":
                c2ws = generate_static(args.num_frames)
            data = {"c2ws": c2ws, "num_frames": args.num_frames, "type": t}
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Generated: {output_path}")
    else:
        if args.type == "orbit":
            c2ws = generate_orbit(args.num_frames)
        elif args.type == "zoom_in":
            c2ws = generate_zoom(args.num_frames, 5.0, 2.0)
        elif args.type == "zoom_out":
            c2ws = generate_zoom(args.num_frames, 2.0, 5.0)
        elif args.type == "pan_left":
            c2ws = generate_pan(args.num_frames, "left")
        elif args.type == "pan_right":
            c2ws = generate_pan(args.num_frames, "right")
        elif args.type == "tilt_up":
            c2ws = generate_tilt(args.num_frames, "up")
        elif args.type == "tilt_down":
            c2ws = generate_tilt(args.num_frames, "down")
        elif args.type == "static":
            c2ws = generate_static(args.num_frames)
        data = {"c2ws": c2ws, "num_frames": args.num_frames, "type": args.type}
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Generated: {args.output}")


if __name__ == "__main__":
    main()
