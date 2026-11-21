"""UDP会话适配；沿用原命令/JSON协议，不依赖Qt。"""
import socket
import threading
import json
import numpy as np
from .protocol import (_shape_for_software_send, _anchor_point_for_software_send,
                       _build_send_obj, RECEIVER_IP, RECEIVER_PORT, SENSOR_TIP_SETBACK_M)

class UdpTransport:
    def __init__(self, runtime, bind_address=('0.0.0.0', 8888), receiver=(RECEIVER_IP, RECEIVER_PORT)):
        """输入RuntimeService和地址，输出无I/O适配器；open才绑定端口。"""
        self.runtime = runtime
        self.entrance_anchor_constraint = runtime.anchor
        self.entrance_anchor_lock = runtime.lock
        self.bind_address = bind_address
        self.receiver = receiver
        self.socket = None
        self.thread = None
        self.running = False
        self.error = None

    @property
    def insert_dis(self):
        """输入无，输出当前推进量float(mm)或None。"""
        return self.runtime.state.insert_value_mm

    @insert_dis.setter
    def insert_dis(self, value):
        """输入推进量float(mm)，输出None，交模块8保存。"""
        self.runtime.set_insertion(value)

    def open(self):
        """输入无，输出None；绑定UDP并启动可关闭的命令接收线程。"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(self.bind_address)
        self.socket.settimeout(.2)
        self.running = True
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        """输入无，输出None；只接收配置命令，不触碰窗口。"""
        while self.running:
            try:
                data, _ = self.socket.recvfrom(1024)
                result = self._handle_udp_datagram(data)
                if not result['ok']:
                    self.error = result['error']
            except socket.timeout:
                continue
            except OSError:
                break

    def publish(self, frame):
        """输入最终RuntimeFrame，输出None；转换m至mm并按原软件协议发送同一最终结果。"""
        shape = _shape_for_software_send(frame.shape.position_m * 1000,
            self.runtime.anchor.enabled, frame.insert_value_mm)
        point = _anchor_point_for_software_send(frame.anchor.anchored_point_m,
            self.runtime.anchor.enabled, frame.insert_value_mm)
        packet = _build_send_obj(frame.end_pose, shape, frame.raw_signals,
                                 frame.shape.diagnostics, anchor_point_mm=point)
        try:
            self.socket.sendto(json.dumps(packet, default=lambda value: value.tolist()).encode(), self.receiver)
        except OSError as exc:
            self.error = exc

    def close(self):
        """输入无，输出None；终止命令接收并关闭UDP，重复关闭安全。"""
        self.running = False
        if self.socket is not None:
            self.socket.close()
        if self.thread is not None:
            self.thread.join(timeout=2.)

    def _get_anchor_constraint_state(self):
        """Return a JSON-serialisable snapshot of the anchor and insertion state."""
        with self.entrance_anchor_lock:
            constraint = self.entrance_anchor_constraint
            anchor = None if constraint.anchor_m is None else constraint.anchor_m.tolist()
            anchor_frame = (
                None
                if constraint.anchor_frame is None
                else constraint.anchor_frame.tolist()
            )
            insert_value = getattr(self, 'insert_dis', None)
            try:
                values = np.asarray(insert_value, dtype=float).reshape(-1)
                insert_value = float(values[0]) if values.size == 1 and np.isfinite(values[0]) else None
            except (TypeError, ValueError):
                insert_value = None
            return {
                'enabled': bool(constraint.enabled),
                'translation_enabled': bool(constraint.translation_enabled),
                'rotation_enabled': bool(constraint.rotation_enabled),
                'entrance_offset_m': float(constraint.entrance_offset_m),
                'anchor_initialized': constraint.anchor_m is not None,
                'anchor_frame_initialized': constraint.anchor_frame is not None,
                'anchor_m': anchor,
                'anchor_frame': anchor_frame,
                'insertion_baseline_mm': constraint.insertion_baseline_mm,
                'last_relative_insert_mm': float(constraint.last_relative_insert_mm),
                'insert_value_mm': insert_value,
            }

    def configure_anchor_constraint(self, enabled=None, entrance_offset_m=None, reset=True):
        """Atomically configure/reset the live anchor and return its actual state."""
        with self.entrance_anchor_lock:
            self.entrance_anchor_constraint.configure(
                enabled=enabled,
                entrance_offset_m=entrance_offset_m,
                reset=reset,
            )
            return self._get_anchor_constraint_state()

    def _dispatch_udp_command(self, response):
        """Validate one decoded UDP command, apply it, and return an internal result."""
        if not isinstance(response, dict):
            raise ValueError("UDP command must be a JSON object")

        cmd = response.get('cmd')
        request_id = response.get('request_id')
        if not isinstance(cmd, str) or not cmd:
            raise ValueError("cmd must be a non-empty string")

        if cmd == 'insert_dis':
            insert_value = response.get('data')
            if type(insert_value) not in (int, float) or not np.isfinite(insert_value):
                raise ValueError("insert_dis data must be one finite number in mm")
            with self.entrance_anchor_lock:
                self.insert_dis = float(insert_value)
            state = self._get_anchor_constraint_state()
        elif cmd == 'anchor_configure':
            payload = response.get('data', {})
            if not isinstance(payload, dict):
                raise ValueError("anchor_configure data must be a JSON object")
            enabled = payload.get('enabled')
            reset = payload.get('reset', True)
            offset = payload.get('entrance_offset_m')
            effective_offset = offset
            if enabled is not None and type(enabled) is not bool:
                raise ValueError("enabled must be a JSON boolean")
            if type(reset) is not bool:
                raise ValueError("reset must be a JSON boolean")
            if offset is not None:
                if type(offset) not in (int, float):
                    raise ValueError("entrance_offset_m must be a JSON number in m")
                if not np.isfinite(offset) or offset < 0.0:
                    raise ValueError("entrance_offset_m must be a finite nonnegative number in m")
                if offset < SENSOR_TIP_SETBACK_M:
                    raise ValueError(
                        "entrance_offset_m must be at least 0.01 m because the sensor "
                        "is 0.01 m behind the catheter tip"
                    )
                effective_offset = max(0.0, float(offset) - SENSOR_TIP_SETBACK_M)
            state = self.configure_anchor_constraint(
                enabled=enabled,
                entrance_offset_m=effective_offset,
                reset=reset,
            )
        elif cmd == 'anchor_reset':
            state = self.configure_anchor_constraint( reset=True)
        elif cmd == 'anchor_status':
            state = self._get_anchor_constraint_state()
        else:
            raise ValueError(f"unknown UDP command: {cmd}")

        return {
            'ok': True,
            'cmd': cmd,
            'request_id': request_id,
            'state': state,
        }

    def _handle_udp_datagram(self, data):
        """Decode a datagram and return an internal result without sending a reply."""
        cmd = None
        request_id = None
        try:
            try:
                response = json.loads(data.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid JSON datagram: {exc}") from exc
            if isinstance(response, dict):
                cmd = response.get('cmd')
                request_id = response.get('request_id')
            return self._dispatch_udp_command( response)
        except Exception as exc:
            return {
                'ok': False,
                'cmd': cmd,
                'request_id': request_id,
                'error': str(exc),
            }

