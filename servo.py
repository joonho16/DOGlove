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
DEVICENAME                  = '/dev/ttyUSB1'
# DEVICENAME                  = '/dev/tty.usbserial-FT88YRMM'

# DEFAULT_POS_SCALE = 2.0 * np.pi / 4096  # 0.088 degrees per unit
DEFAULT_POS_SCALE = 2.0 * 180 / 4096  # 0.088 degrees per unit
#********* DYNAMIXEL Model definition *********

# Force feedback settings
MODE_CURRENT_BASED_POSITION = 5
MA_TO_GRAM = 1.13               # mA → gram 변환 계수 (Kt / L_finger / g, URDF 기반 근사)
FORCE_THRESHOLD_G = 50          # gram - force feedback 시작 threshold
FORCE_MAX_G = 3000              # gram - maximum force (KP 최대)
KP_MAX = 800                    # Maximum Position P Gain
SERVO_CURRENT_LIMIT = 150       # mA - DOGlove servo current limit (safety)

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

        # Force feedback state
        self.force_enabled = False
        self.force_listener = None
        self.current_kp = {dxl_id: 0 for dxl_id in DXL_ID}
        self.present_positions = {dxl_id: 0 for dxl_id in DXL_ID}

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

        for bh_name, dxl_id in FORCE_MAPPING.items():
            current_ma = abs(effort.get(bh_name, 0))
            force_g = current_ma * MA_TO_GRAM

            if force_g < FORCE_THRESHOLD_G:
                kp = 0
            else:
                ratio = min(1.0, (force_g - FORCE_THRESHOLD_G) / (FORCE_MAX_G - FORCE_THRESHOLD_G))
                kp = int(KP_MAX * ratio)

            # Only write if changed (reduce bus traffic)
            if kp != self.current_kp[dxl_id]:
                self.current_kp[dxl_id] = kp
                self.packetHandler.write2ByteTxRx(self.portHandler, dxl_id, ADDR_POSITION_P_GAIN, kp)

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

        print(f"Servo joint angles: {servo_joint_angles}")

        # Send the first voltage value via UDP
        # message = struct.pack("f", voltages[0])
        format_string = "f" * len(servo_joint_angles)
        # Pack all the values in the list
        message = struct.pack(format_string, *servo_joint_angles)
        self.udp_socket.sendto(message, (udp_ip, udp_port_servo))

        # Force feedback update
        self.update_force_feedback()

    def start(self):
        print("Servo reader started")
        self.thread = threading.Thread(target=self.read_from_uart)
        self.thread.start()

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
