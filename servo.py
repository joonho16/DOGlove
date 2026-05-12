import socket
import struct
import threading
import time

import numpy as np

from dynamixel_sdk import *  # Uses Dynamixel SDK library

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

#********* DYNAMIXEL Model definition *********
# Control table address
ADDR_OPERATING_MODE         = 11
ADDR_CURRENT_LIMIT          = 38
LEN_CURRENT_LIMIT           = 2
ADDR_TORQUE_ENABLE          = 64
ADDR_LED_RED                = 65
LEN_LED_RED                 = 1         # Data Byte Length
ADDR_POSITION_P_GAIN        = 84
LEN_POSITION_P_GAIN         = 2
ADDR_GOAL_POSITION          = 116
LEN_GOAL_POSITION           = 4         # Data Byte Length
ADDR_PRESENT_POSITION       = 132
LEN_PRESENT_POSITION        = 4         # Data Byte Length
DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the Minimum Position Limit of product eManual
DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the Maximum Position Limit of product eManual
BAUDRATE                    = 3000000

# DYNAMIXEL Protocol Version 2.0
# https://emanual.robotis.com/docs/en/dxl/protocol2/
PROTOCOL_VERSION            = 2.0

# Make sure that each DYNAMIXEL ID should have unique ID.
DXL_ID                     = [0, 1, 2, 3, 4]                 # Dynamixe ID

# Use the actual port assigned to the U2D2.
# ex) Windows: "COM*", Linux: "/dev/ttyUSB*", Mac: "/dev/tty.usbserial-*"
DEVICENAME                  = '/dev/ttyUSB2'
# DEVICENAME                  = '/dev/tty.usbserial-FT88YRMM'

# DEFAULT_POS_SCALE = 2.0 * np.pi / 4096  # 0.088 degrees per unit
DEFAULT_POS_SCALE = 2.0 * 180 / 4096  # 0.088 degrees per unit
#********* DYNAMIXEL Model definition *********

# Force feedback settings
MODE_CURRENT_BASED_POSITION = 5
EFFORT_THRESHOLD = 50           # effort - force feedback 시작 threshold
EFFORT_LRA_MAX = 100            # effort - LRA only 상한, 이상이면 KP도 적용
EFFORT_MAX = 3000               # effort - maximum (KP 최대)
KP_MAX = 800                    # Maximum Position P Gain
SERVO_CURRENT_LIMIT = 150       # mA - DOGlove servo current limit (safety)
DEBOUNCE_TIME = 0.1             # seconds - threshold 넘은 후 이 시간 유지해야 동작

# bluehand FE joint name -> DOGlove servo ID
FORCE_MAPPING = {
    'finger1_FE': 0,   # thumb
    'finger2_FE': 1,   # index
    'finger3_FE': 2,   # middle
    'finger4_FE': 3,   # ring
}

data_length = 5  # Number of data values in the block

# Configure UDP settings
udp_ip = "127.0.0.1"  # Localhost IP
udp_port_servo = 5010  # Port to send data to
udp_port_lra = 5012    # Port for LRA force data


class ForceListener(Node):
    """ROS2 node that listens to bluehand joint_states for force feedback."""
    def __init__(self):
        super().__init__('doglove_force_listener')
        self.latest_effort = {}
        self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)

    def joint_state_callback(self, msg):
        for i, name in enumerate(msg.name):
            if i < len(msg.effort):
                self.latest_effort[name] = msg.effort[i]


class ServoReader:
    def __init__(self):
        self.running = True
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.lra_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Force feedback state
        self.force_enabled = False
        self.force_listener = None
        self.current_kp = {dxl_id: 0 for dxl_id in DXL_ID}
        self.present_positions = {dxl_id: 0 for dxl_id in DXL_ID}
        self.effort_above_since = {dxl_id: None for dxl_id in DXL_ID}  # debounce timestamp

        # Servo offset calibration
        self.servo_offsets = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.latest_servo_angles = [180.0, 180.0, 180.0, 180.0, 180.0]
        self.offsets_set = False

        # Open the port
        if self.portHandler.openPort():
            print("Succeeded to open the port!")
        else:
            print("Failed to open the port!")
            self.stop()
            return

        # Set port baudrate
        if self.portHandler.setBaudRate(BAUDRATE):
            print(f"Succeeded to change the baudrate to {BAUDRATE}")
        else:
            print(f"Failed to change the baudrate to {BAUDRATE}")
            self.stop()
            return

        # Initialize the GroupBulkWrite instance
        self.groupBulkWrite = GroupBulkWrite(self.portHandler, self.packetHandler)
        # Initialize the GroupBulkRead instance
        self.groupBulkRead = GroupBulkRead(self.portHandler, self.packetHandler)

    def init_force_feedback(self):
        """Initialize DOGlove servos for force feedback."""
        print("Initializing force feedback...")

        # 1) Disable torque (required to change operating mode)
        for dxl_id in DXL_ID:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)

        # 2) Set operating mode: current-based position control
        for dxl_id in DXL_ID:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_OPERATING_MODE, MODE_CURRENT_BASED_POSITION)

        # 3) Enable torque
        for dxl_id in DXL_ID:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)

        # 4) Set current limit (safety) and initial KP = 0 (free movement)
        for dxl_id in DXL_ID:
            self.packetHandler.write2ByteTxRx(self.portHandler, dxl_id, ADDR_CURRENT_LIMIT, SERVO_CURRENT_LIMIT)
            self.packetHandler.write2ByteTxRx(self.portHandler, dxl_id, ADDR_POSITION_P_GAIN, 0)

        # 5) Set goal position = present position
        for dxl_id in DXL_ID:
            pos, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, dxl_id, ADDR_PRESENT_POSITION)
            if pos > 0x7FFFFFFF:
                pos -= 4294967296
            self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_GOAL_POSITION, pos & 0xFFFFFFFF)

        # 6) Start ROS2 force listener
        rclpy.init()
        self.force_listener = ForceListener()
        self.ros_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.ros_thread.start()

        self.force_enabled = True
        print("Force feedback initialized!")

    def _spin_ros(self):
        """Spin ROS2 node in background thread."""
        while self.running and rclpy.ok():
            rclpy.spin_once(self.force_listener, timeout_sec=0.01)

    def update_force_feedback(self):
        """Map bluehand FE current -> DOGlove KP gain."""
        if not self.force_enabled or self.force_listener is None:
            return

        effort = self.force_listener.latest_effort

        now = time.time()
        force_values = []
        for bh_name, dxl_id in FORCE_MAPPING.items():
            eff = abs(effort.get(bh_name, 0))

            # Debounce: threshold 넘은 시점 기록, 일정 시간 유지해야 동작
            if eff >= EFFORT_THRESHOLD:
                if self.effort_above_since[dxl_id] is None:
                    self.effort_above_since[dxl_id] = now
                active = (now - self.effort_above_since[dxl_id]) >= DEBOUNCE_TIME
            else:
                self.effort_above_since[dxl_id] = None
                active = False

            if not active:
                kp = 0
                force_values.append(0.0)
            elif eff < EFFORT_LRA_MAX:
                # 50-100: LRA only, no KP
                kp = 0
                force_values.append(eff)
            else:
                # > 100: KP proportional
                ratio = min(1.0, (eff - EFFORT_LRA_MAX) / (EFFORT_MAX - EFFORT_LRA_MAX))
                kp = int(KP_MAX * ratio)
                force_values.append(eff)

            # Only write if changed (reduce bus traffic)
            if kp != self.current_kp[dxl_id]:
                self.current_kp[dxl_id] = kp
                self.packetHandler.write2ByteTxRx(self.portHandler, dxl_id, ADDR_POSITION_P_GAIN, kp)

        # LRA용 effort 값을 UDP로 전송 (4 fingers: thumb, index, middle, ring)
        lra_msg = struct.pack("f" * 4, *force_values)
        self.lra_socket.sendto(lra_msg, (udp_ip, udp_port_lra))

        # Update goal position = present position when KP is 0 (free movement)
        # When KP > 0 (force active), hold position for spring-like resistance
        for dxl_id in DXL_ID:
            if self.current_kp.get(dxl_id, 0) == 0:
                pos = self.present_positions.get(dxl_id, 0)
                if pos < 0:
                    pos += 4294967296
                self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_GOAL_POSITION, pos & 0xFFFFFFFF)

    def read_from_uart(self):
        while self.running:
            # print("Reading from UART...")
            self.process_buffer()

    def process_buffer(self):
        self.process_block()

    def process_block(self):
        # Bulkread present position and LED status
        dxl_comm_result = self.groupBulkRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        servo_joint_angles = [180.0, 180.0, 180.0, 180.0, 180.0]
        for id in DXL_ID:
            # Check if groupbulkread data of Dynamixel_id is available
            dxl_getdata_result = self.groupBulkRead.isAvailable(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            if dxl_getdata_result != True:
                print("[ID:%03d] groupBulkRead getdata failed" % id)
            else:
                # Get present position value
                dxl_present_position = self.groupBulkRead.getData(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                if dxl_present_position > 0x7FFFFFFF:
                    dxl_present_position -= 4294967296
                self.present_positions[id] = dxl_present_position
                # print("[ID:%03d] Present Position: %f" % (id, dxl_present_position*DEFAULT_POS_SCALE))
                servo_joint_angles[id] = dxl_present_position*DEFAULT_POS_SCALE

        self.latest_servo_angles = list(servo_joint_angles)

        # Apply offsets if calibrated
        if self.offsets_set:
            offset_applied = [servo_joint_angles[i] - self.servo_offsets[i] for i in range(5)]
            print(f"Servo joint angles (offset): {[round(v, 2) for v in offset_applied]}")
        else:
            offset_applied = servo_joint_angles
            print(f"Servo joint angles: {servo_joint_angles}")

        # Send via UDP
        format_string = "f" * len(offset_applied)
        message = struct.pack(format_string, *offset_applied)
        self.udp_socket.sendto(message, (udp_ip, udp_port_servo))

        # Force feedback update
        self.update_force_feedback()

    def input_listener(self):
        """Listen for 'y' input to capture current servo angles as offsets."""
        while self.running:
            try:
                user_input = input("Press 'y' + Enter to set current servo angles as offset: ")
                if user_input.strip().lower() == 'y':
                    self.servo_offsets = list(self.latest_servo_angles)
                    self.offsets_set = True
                    print(f">>> Offsets set: {[round(v, 2) for v in self.servo_offsets]}")
            except EOFError:
                break

    def start(self):
        print("Servo reader started")
        self.thread = threading.Thread(target=self.read_from_uart)
        self.thread.start()

        self.input_thread = threading.Thread(target=self.input_listener, daemon=True)
        self.input_thread.start()

        # Add parameter storage for Dynamixel present position and LED status
        for id in DXL_ID:
            dxl_addparam_result = self.groupBulkRead.addParam(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkRead addparam failed" % id)
                self.stop()
                return

    def stop(self):
        print("Closing UART port...")
        self.running = False
        self.thread.join()

        # Disable force feedback: set KP = 0 and disable torque
        if self.force_enabled:
            for dxl_id in DXL_ID:
                self.packetHandler.write2ByteTxRx(self.portHandler, dxl_id, ADDR_POSITION_P_GAIN, 0)
                self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
            self.force_listener.destroy_node()
            rclpy.shutdown()

        self.groupBulkRead.clearParam()
        self.groupBulkWrite.clearParam()
        self.portHandler.closePort()

        self.udp_socket.close()

if __name__ == "__main__":
    reader = ServoReader()
    try:
        reader.start()
        reader.init_force_feedback()
        print("Servo port opened successfully (with force feedback)")

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Program interrupted by user")

    finally:
        reader.stop()
