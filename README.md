# p1d2_khushei_meda

ROS1 package for drawing an M-shaped trajectory in `turtlesim` using closed-loop pose feedback.

## Package Contents

```text
p1d2_khushei_meda/
├── CMakeLists.txt
├── package.xml
└── scripts/
    └── p1d2_khushei_meda
```

Place the package at:

```text
<catkin_ws>/src/p1d2_khushei_meda
```

## Prerequisites

- Ubuntu with ROS1 Noetic
- `turtlesim` installed
- Catkin workspace

## Build

```bash
source /opt/ros/noetic/setup.bash
cd <catkin_ws>
catkin_make
source devel/setup.bash
```

## Run

Use 3 terminals.

Terminal 1:

```bash
source /opt/ros/noetic/setup.bash
roscore
```

Terminal 2:

```bash
source /opt/ros/noetic/setup.bash
rosrun turtlesim turtlesim_node
```

Terminal 3:

```bash
source /opt/ros/noetic/setup.bash
source <catkin_ws>/devel/setup.bash
rosrun p1d2_khushei_meda p1d2_khushei_meda
```

## What The Node Does

- Subscribes to `/turtle1/pose` for feedback
- Publishes `/turtle1/cmd_vel` to drive the turtle
- Uses `/clear`, `/turtle1/set_pen`, and `/turtle1/teleport_absolute` services
- Logs each target waypoint and total completion time
- Stops if runtime exceeds timeout (default 120 seconds)

## Runtime 

The drawing is completed in ~74.5 sec.

### Parameters Defaults:
- `~k_lin` = `2.0`
- `~k_ang` = `8.5`
- `~v_max` = `1.8`
- `~w_max` = `8.0`
- `~dist_tol` = `0.02`
- `~heading_tol` = `0.006`
- `~heading_slow_tol` = `0.05`
- `~timeout_s` = `120.0`
- `~rate_hz` = `100.0`
- `~segment_settle_s` = `0.03`
- `~pen_r` = `0`, `~pen_g` = `0`, `~pen_b` = `255`
- `~pen_width` = `2`

## AI Disclosure

Used GPT-5.3-Codex to help with writing the program
