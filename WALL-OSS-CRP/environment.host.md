# Host environment — historical development-host snapshot

Captured: 2026-10-22T07:57:05.234421+00:00

> 文档性质：2026-10-22 开发机原始测量，适用范围于 2026-10-29 标注；以下终端输出完整保留。这里的 RTX 3060 Ti、8192 MiB、驱动 580.178.04 和 CUDA 13.0 仅属于当时开发机，不是推理目标主机或 H100 服务器的实测配置。
>
> 当前入口见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。独立推理目标为用户提供的 Ubuntu RTX 5070 Ti（16303 MiB，驱动 595.91.07）；H100 训练服务器的 8 卡资源来自用户确认。两台目标主机均不能引用本文件作为验收证据，部署步骤见 [推理部署说明](WALL_RLT_Inference_Deployment.md)。

```text
$ uname -a
Linux 1225 6.8.0-134-generic #134-Ubuntu SMP PREEMPT_DYNAMIC Fri Jun 26 18:43:11 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux

exit_code=0
```

```text
$ cat /etc/os-release
PRETTY_NAME="Ubuntu 24.04.4 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.4 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=noble
LOGO=ubuntu-logo

exit_code=0
```

```text
$ lscpu
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           39 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  20
On-line CPU(s) list:                     0-19
Vendor ID:                               GenuineIntel
Model name:                              12th Gen Intel(R) Core(TM) i7-12700F
CPU family:                              6
Model:                                   151
Thread(s) per core:                      2
Core(s) per socket:                      12
Socket(s):                               1
Stepping:                                2
CPU(s) scaling MHz:                      44%
CPU max MHz:                             4900.0000
CPU min MHz:                             800.0000
BogoMIPS:                                4224.00
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb intel_pt sha_ni xsaveopt xsavec xgetbv1 xsaves split_lock_detect user_shstk avx_vnni dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp hwp_pkg_req hfi vnmi umip pku ospke waitpkg gfni vaes vpclmulqdq rdpid movdiri movdir64b fsrm md_clear serialize arch_lbr ibt flush_l1d arch_capabilities ibpb_exit_to_user
Virtualization:                          VT-x
L1d cache:                               512 KiB (12 instances)
L1i cache:                               512 KiB (12 instances)
L2 cache:                                12 MiB (9 instances)
L3 cache:                                25 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-19
Vulnerability Gather data sampling:      Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Reg file data sampling:    Mitigation; Clear Register File
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; PBRSB-eIBRS SW sequence; BHI BHI_DIS_S
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Mitigation; IBPB before exit to userspace

exit_code=0
```

```text
$ nvidia-smi
Thu Oct 22 15:57:05 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.178.04             Driver Version: 580.178.04     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060 Ti     Off |   00000000:01:00.0  On |                  N/A |
|  0%   43C    P8             10W /  200W |     565MiB /   8192MiB |     19%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2888      G   /usr/lib/xorg/Xorg                      205MiB |
|    0   N/A  N/A            3104      G   /usr/bin/gnome-shell                     51MiB |
|    0   N/A  N/A            5250      G   /opt/QQ/qq                              126MiB |
|    0   N/A  N/A           16007      G   ...rack-uuid=3190708988185955192         87MiB |
|    0   N/A  N/A           17023      G   /usr/bin/nautilus                        14MiB |
|    0   N/A  N/A           24513      G   /usr/bin/gnome-control-center            17MiB |
+-----------------------------------------------------------------------------------------+

exit_code=0
```

```text
$ nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.free,compute_cap --format=csv
name, uuid, driver_version, memory.total [MiB], memory.free [MiB], compute_cap
NVIDIA GeForce RTX 3060 Ti, GPU-59e5a2a2-5a16-7061-9a1f-c65b3cfeb3de, 580.178.04, 8192 MiB, 7267 MiB, 8.6

exit_code=0
```

```text
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0

exit_code=0
```

```text
$ gcc --version
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

exit_code=0
```

```text
$ free -h
               total        used        free      shared  buff/cache   available
Mem:            15Gi       5.9Gi       3.6Gi       233Mi       6.5Gi       9.5Gi
Swap:             0B          0B          0B

exit_code=0
```

```text
$ df -h /home/pangk/workspaces/WALL-OSS-CRP
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda3       492G  400G   67G  86% /

exit_code=0
```

```text
$ /home/pangk/workspaces/WALL-OSS-CRP/env/bin/python --version
Python 3.10.22

exit_code=0
```
