# 倒立摆计算机控制教学平台（Inverted_Pendulum）

一套面向课堂的**计算机控制系统**教学软件，对象是学生自制的“步进电机 + 同步带 + 小车 + 电位器 + 钢管摆杆”一级倒立摆。平台由三部分组成：

1. **数字孪生仿真**：非线性模型，含采样、延迟、量化噪声、驱动器滞后、皮带间隙、导轨端点等非理想因素。可以实时或慢动作演示 PD、串级 PID、LQR、极点配置和能量起摆。
2. **闭环分析**：给出增益、z 平面闭环极点和延迟裕度。线性理论和仿真结果并列展示。
3. **实物实时控制**：PC 以 200 Hz 读取电位器角度，经 PD42S1 闭环步进驱动器的**通信速度模式**驱动小车。支持影子模式（只读、电机不动）、闭环模式，以及不需要硬件的软件在环（SIL）演示。

> **当前状态：请务必先读这一段**
> - 仿真、分析、GUI、全部硬件代码路径都已完成，52 项自动测试全部通过。硬件路径包括串口解析、PD42S1 帧编解码、电机线程、实时循环和安全监督。
> - PD42S1 帧格式 `C5 | 地址 | 功能码 | 数据 | sum8 | 5C` 已与原厂上位机日志中的 10 帧**逐字节核对**。
> - **速度指令 `set_speed` 与读位置 `read_position` 的功能码和数据布局尚未填写。** 手头没有 PD42S1 手册的指令表，软件不会猜测。填写并验证之前，所有会让电机运动的命令都会被拒绝。操作步骤见 [docs/07](docs/07_PD42S1通信协议适配.md)。
> - 本软件**尚未在您的实物上通电运动验证**。实物闭环前必须按 [docs/05](docs/05_实物调试流程.md) 逐步完成标定和安全检查。

## 1 分钟上手（Windows，Conda 或普通 Python ≥ 3.9，推荐 3.11+）

在工程根目录打开终端（Anaconda Prompt 或 PowerShell）：

```bat
python -m pip install -e .
python -m pendulum_lab gui
```

- 可执行命令叫 `pendulum-lab`（连字符），Python 模块叫 `pendulum_lab`（下划线）。`python -m pendulum_lab ...` 和 `pendulum-lab ...` 两种写法等价。
- GUI 完全离线、不碰串口，可直接课堂演示。

常用命令：

```bat
python -m pendulum_lab params                       :: 由测量数据推出的物理参数
python -m pendulum_lab design --png poles.png       :: 各算法增益、闭环极点、延迟裕度
python -m pendulum_lab compare --kick 5:0.5 --png cmp.png   :: 同一场景对比 PD/PID/LQR/极点配置
python -m pendulum_lab simulate --algo swingup --duration 15 --png swing.png
python -m pendulum_lab sil --algo lqr               :: 用"模拟实物"跑一遍实时硬件代码路径
python -m pendulum_lab diagnose --sensor COM12      :: 只读：角度传感器速率/噪声/格式
python -m pendulum_lab probe --motor COM11 --rate-hz 5 --count 100 --csv data\probe.csv   :: 只读：驱动器通信误码率
python -m pytest -q                                 :: 52 项自动测试
```

## 由您的测量得到的关键物理结论

| 量 | 数值 | 来源 |
|---|---|---|
| 摆杆总质量 $m$ | 105.4 g（钢管 86.3 g + 铝配重 19.1 g） | 几何（Ø8×1 mm 钢管 50 cm，支点距下端 12 cm） |
| 质心到支点 $l_c$ | 17.3 cm | 几何 |
| 对支点转动惯量 $J$ | $5.80\times10^{-3}\,\mathrm{kg\,m^2}$ | 几何 |
| 小角度周期 $T_0$ | 几何 1.1330 s；实测 1.1265–1.1317 s | 20 次 24.85 s，按大振幅周期修正 |
| $\omega_0=\sqrt{mgl_c/J}$ | 5.55 rad/s | 两者相差 < 0.6 %，印证支点位置假设 |
| 开环不稳定极点 | $p\approx+5.53\ \mathrm{s^{-1}}$（倍增时间 125 ms） | $\sqrt{\omega_0^2}$ |
| 等效摆长 $L_{eq}=g/\omega_0^2$ | 31.8 cm | 运动学驱动下唯一重要的参数 |
| 电位器斜率 | 676.7 LSB/rad（0.0847°/LSB） | (4092−1966)/π，与手册 345°/4095 相差 0.5 % |
| 电位器死区 | **正好在下垂位置旁边** | 下垂读数 4091–4093 已到行程末端 |

一个比数值本身更重要的结论：步进电机刚性驱动小车时，控制输入是小车**加速度**。摆杆动力学完全由 $\omega_0$ 决定（再加上很小的阻尼项），用秒表就能测准，学生可以亲手完成“建模—辨识—验证”的闭环。详见 [docs/02](docs/02_物理建模与参数辨识.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [01 系统总体设计](docs/01_系统总体设计.md) | 架构、控制层次、线程/进程、时序预算、运行模式 |
| [02 物理建模与参数辨识](docs/02_物理建模与参数辨识.md) | 拉格朗日推导、运动学驱动模型、几何参数、摆动试验辨识、传感器模型 |
| [03 控制算法原理](docs/03_控制算法原理.md) | PD、串级 PID、LQR、极点配置、卡尔曼滤波与零偏估计、能量起摆、离散化与延迟 |
| [04 软件使用手册](docs/04_软件使用手册.md) | 安装、全部命令、GUI、配置文件字段、日志格式 |
| [05 实物调试流程](docs/05_实物调试流程.md) | 从只读诊断到首次闭环的逐步清单与通过标准 |
| [06 教学实验指导书](docs/06_教学实验指导书.md) | 10 个课堂实验（仿真 + 实物），含思考题 |
| [07 PD42S1 通信协议适配](docs/07_PD42S1通信协议适配.md) | 帧格式、从手册填写指令表、对照原厂日志验证 |
| [08 常见问题与故障排查](docs/08_常见问题与故障排查.md) | 命令拼写、串口、4.47 V、漂移、振荡、极限环等 |

## 工程结构

```
pendulum_lab/
  model/      params.py 物理参数 | dynamics.py 非线性/线性模型 | identification.py 摆动辨识
  control/    discrete.py 离散化+延迟+DLQR+Ackermann+Kalman | controllers.py PD/PID/LQR/极点配置/起摆
              estimators.py 微分/卡尔曼(零偏)/起摆估计 | analysis.py 闭环极点、延迟裕度
              actuator.py 加速度→速度指令 | safety.py 安全监督（仿真与实物共用）
  sim/        simulator.py 数字孪生
  hw/         pd42s1.py 帧编解码+指令表 | sensor.py 角度串口 | motor.py 电机线程 | runtime.py 实时循环
              process.py GUI 独立控制进程 | timing.py 高精度定时 | fake.py 模拟实物(SIL)
  tools/      hwtools.py 诊断/标定/阶跃测试 | plotting.py 作图
  gui/app.py  Tkinter 图形界面
  cli.py      命令行
config/       pd42s1_protocol.json 驱动器指令表 | physical_config.example.json 实物配置模板
docs/         说明文档与图
examples/     make_teaching_figures.py 重新生成文档配图
tests/        自动测试（含对原厂日志帧的逐字节核对、SIL 实时测试）
```

## 安全

- 12 V 电源开关就是急停，必须放在操作者手边。建议串接蘑菇头急停。
- **通信中断时驱动器会保持最后的速度指令**（速度模式的固有特性）。PC 死机、USB 被拔都可能让小车冲向端点，所以两端必须有**独立于 PC 的硬件限位**，能切断电源或使能。见 docs/05 第 0 步。
- 软件停车方式是速度指令斜坡减到 0。`FC` 刹车会锁存，只作最后手段。
