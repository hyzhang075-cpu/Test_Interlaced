# 入口锚定 UDP 外部控制协议

## 1. 网络与编码

- Demonstrator 监听：`0.0.0.0:8888/UDP`。
- 报文编码：UTF-8 JSON，每个 UDP datagram 对应一条完整命令。
- 距离单位：`entrance_offset_m` 为 m；`insert_dis` 为 mm。
- 控制命令执行后，Demonstrator 会向该命令的 UDP 源地址回发 ACK/NACK JSON。
- Demonstrator 本机命令行也会打印成功或失败摘要，但命令行不是机器接口；外部软件应读取 UDP 回包。

`request_id` 由调用方生成，Demonstrator 原样回传，用于匹配请求和响应。

## 2. 启动 Demonstrator

默认入口距离为 0.06 m。可在启动时启用锚定：

```powershell
python main.py --task demo --anchor_constraint_enabled --anchor_offset_m 0.06
```

也可以不在启动时启用，待程序运行后使用 `anchor_configure` 开启。

## 3. 配置并重新锚定

发送：

```json
{
  "cmd": "anchor_configure",
  "request_id": "cfg-0001",
  "data": {
    "enabled": true,
    "entrance_offset_m": 0.07,
    "reset": true
  }
}
```

等价进程内调用：

```python
demo.configure_anchor_constraint(
    enabled=True,
    entrance_offset_m=0.07,
    reset=True,
)
```

约束：

- `enabled` 必须是 JSON boolean，不能传字符串 `"true"`/`"false"`。
- `entrance_offset_m` 必须是有限非负数，单位 m。
- `reset` 必须是 JSON boolean；省略时默认为 `true`。
- 修改 `entrance_offset_m` 时推荐始终使用 `reset=true`。

`reset=true` 会清除固定入口坐标、推进量零点和上次相对推进量。下一帧重构曲线建立新入口坐标；当前最新的有限非零机械臂读数会在下一帧成为新推进量零点。

## 4. 仅 reset 当前锚定会话

发送：

```json
{
  "cmd": "anchor_reset",
  "request_id": "reset-0001"
}
```

该命令保留当前 `enabled` 和 `entrance_offset_m`，只清空：

- `anchor_m`；
- `insertion_baseline_mm`；
- `last_relative_insert_mm`。

## 5. 查询实际状态

发送：

```json
{
  "cmd": "anchor_status",
  "request_id": "status-0001"
}
```

成功回包示例：

```json
{
  "ok": true,
  "cmd": "anchor_status",
  "request_id": "status-0001",
  "state": {
    "enabled": true,
    "entrance_offset_m": 0.07,
    "anchor_initialized": true,
    "anchor_m": [0.13, 0.0, 0.0],
    "insertion_baseline_mm": 156.8,
    "last_relative_insert_mm": 10.0,
    "insert_value_mm": 166.8
  }
}
```

配置/reset 后立即返回的 `anchor_initialized=false` 是正常状态：新固定入口要等下一帧曲线才能计算。

## 6. 推进量输入

保留原协议：

```json
{
  "cmd": "insert_dis",
  "request_id": "insert-0001",
  "data": 166.8
}
```

`data` 必须是一个有限数，单位 mm。前导零或后续零由锚定状态机视为占位/掉线值，不会改变推进零点或最后有效相对推进量。

## 7. 失败回包

非法 JSON、未知命令、错误类型、NaN、无穷值或负入口距离不会终止 UDP 接收线程，并返回：

```json
{
  "ok": false,
  "cmd": "anchor_configure",
  "request_id": "cfg-0002",
  "error": "enabled must be a JSON boolean"
}
```

失败配置不会部分修改原有锚定状态。

## 8. Python 调用示例

```python
import json
import socket

DEMO_ADDRESS = ("192.168.1.100", 8888)  # 替换为运行 demo 的电脑 IP

command = {
    "cmd": "anchor_configure",
    "request_id": "cfg-0001",
    "data": {
        "enabled": True,
        "entrance_offset_m": 0.07,
        "reset": True,
    },
}

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.settimeout(2.0)
    sock.sendto(json.dumps(command).encode("utf-8"), DEMO_ADDRESS)
    response_data, response_address = sock.recvfrom(4096)
    response = json.loads(response_data.decode("utf-8"))
    print(response_address, response)
```

生产接入建议：调用方设置 1–2 秒超时。当前 `request_id` 只用于请求/响应匹配，不提供服务端去重；如果 `anchor_configure` 或 `anchor_reset` 的 ACK 超时，不要盲目重发，因为首个命令可能已经执行且新锚点可能已经建立。应先发送新的 `anchor_status` 查询实际状态，再由上层决定是否重新配置。UDP 可能丢包、乱序或重复，因此 reset/configure 命令不得并发发送。
