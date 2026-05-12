from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import rclpy
from sensor_msgs.msg import JointState

from glove_mcu import UDPReceiver

_HERE = Path(__file__).parent
_XML = _HERE / "DOGlove_meshes" / "DOGlove-v3.xml"
_BLUEHAND_XML = _HERE / "hand_urdf_description" / "meshes" / "bluehand.xml"


def create_combined_xml():
    """Combine DOGlove and bluehand XMLs into a single MuJoCo model."""
    doglove_tree = ET.parse(str(_XML))
    bluehand_tree = ET.parse(str(_BLUEHAND_XML))

    doglove_root = doglove_tree.getroot()
    bluehand_root = bluehand_tree.getroot()

    # Convert DOGlove mesh file paths to absolute
    doglove_meshdir = str(_XML.parent)
    for asset in doglove_root.findall("asset"):
        for mesh in asset.findall("mesh"):
            f = mesh.get("file")
            if f and not Path(f).is_absolute():
                mesh.set("file", doglove_meshdir + "/" + f)

    # Find the last <asset> section in DOGlove to append bluehand meshes
    asset_sections = doglove_root.findall("asset")
    asset_section = asset_sections[-1]

    # Add bluehand meshes with "bh_" prefix and absolute file paths
    bluehand_meshdir = str(_BLUEHAND_XML.parent)
    for asset in bluehand_root.findall("asset"):
        for mesh in asset.findall("mesh"):
            new_mesh = ET.SubElement(asset_section, "mesh")
            for k, v in mesh.attrib.items():
                if k == "name":
                    new_mesh.set(k, "bh_" + v)
                elif k == "file":
                    new_mesh.set(k, bluehand_meshdir + "/" + v)
                else:
                    new_mesh.set(k, v)

    # Add bluehand worldbody content under a container body with offset
    doglove_wb = doglove_root.find("worldbody")
    bh_container = ET.SubElement(doglove_wb, "body")
    bh_container.set("name", "bluehand_root")
    bh_container.set("pos", "0.3 0 0.3")
    bh_container.set("euler", "0 -1.5708 -1.5708")  # 위에서 봤을 때 시계방향 90도 회전

    def prefix_bluehand(elem, parent):
        """Recursively copy bluehand worldbody elements with prefixed names."""
        new_elem = ET.SubElement(parent, elem.tag)
        for k, v in elem.attrib.items():
            if k in ("name", "body1", "body2"):
                new_elem.set(k, "bh_" + v)
            elif k == "mesh":
                new_elem.set(k, "bh_" + v)
            elif k == "joint":
                new_elem.set(k, "bh_" + v)
            else:
                new_elem.set(k, v)
        new_elem.text = elem.text
        new_elem.tail = elem.tail
        for child in elem:
            prefix_bluehand(child, new_elem)

    bluehand_wb = bluehand_root.find("worldbody")
    for child in bluehand_wb:
        prefix_bluehand(child, bh_container)

    return ET.tostring(doglove_root, encoding="unicode")



def coupling_pip(mcp):
    """MCP → PIP coupling (4th degree polynomial)"""
    return -0.1108*mcp**4 + 0.3230*mcp**3 - 0.2964*mcp**2 + 1.2475*mcp + 0.0034

def coupling_dip(mcp):
    """MCP → DIP coupling (5th degree polynomial)"""
    return 0.1005*mcp**5 + 0.3467*mcp**4 - 0.4016*mcp**3 + 0.3414*mcp**2 + 0.6245*mcp - 0.00087


def get_body_pos(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    body_pos = data.xpos[body_id]
    return body_pos

def get_site_pos(model, data, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    site_pos = data.site_xpos[site_id]
    return site_pos

def find_actuator_for_joint(model, joint_name):
    """Finds the actuator controlling the given joint."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    
    # Iterate through actuators to find the one controlling this joint
    for actuator_id in range(model.nu):  # `nu` is the number of actuators
        # `actuator_trnid` gives the joint or tendon that each actuator controls
        # actuator_trnid[actuator_id, 0] contains the joint/tendon ID
        if model.actuator_trnid[actuator_id][0] == joint_id:
            return actuator_id
    return None


def main():
    combined_xml = create_combined_xml()
    model = mujoco.MjModel.from_xml_string(combined_xml)
    data = mujoco.MjData(model)

    receiver = UDPReceiver()
    receiver.start()

    # --- ROS2 publisher setup ---
    rclpy.init()
    ros_node = rclpy.create_node('doglove_bluehand')
    goal_pub = ros_node.create_publisher(JointState, 'goal_joint_states', 10)

    # --- DOGlove FK setup ---
    doglove_joint_names = [
        'thumb_bend_1', 'thumb_bend_2', 'thumb_split', 'thumb_mcp',
        'index_bend_1', 'index_bend_2', 'index_split',
        'middle_bend_1', 'middle_bend_2', 'middle_split',
        'ring_bend_1', 'ring_bend_2', 'ring_split',
        'pinky_bend_1', 'pinky_bend_2', 'pinky_split',
        'thumb_bend_3', 'index_bend_3', 'middle_bend_3', 'ring_bend_3', 'pinky_bend_3',
    ]
    doglove_joint_ids = []
    doglove_qpos_indices = []
    for name in doglove_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        doglove_joint_ids.append(jid)
        doglove_qpos_indices.append(model.jnt_qposadr[jid])

    # --- AA joint 직접 매핑: DOGlove split → bluehand aa (엄지 제외) ---
    aa_mapping = [
        ('thumb_split', 'bh_aa1'),
        ('index_split', 'bh_aa2'),
        ('middle_split', 'bh_aa3'),
        ('ring_split', 'bh_aa4'),
    ]
    aa_src_qpos = []  # DOGlove split joint qpos index
    aa_dst_qpos = []  # bluehand aa joint qpos index
    for src, dst in aa_mapping:
        src_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, src)
        dst_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dst)
        aa_src_qpos.append(model.jnt_qposadr[src_jid])
        aa_dst_qpos.append(model.jnt_qposadr[dst_jid])

    # --- FE joint 매핑: DOGlove bend_3 → bluehand mcp, coupling → pip/dip/act (엄지 제외) ---
    fe_mapping = [
        ('thumb_bend_3',  'bh_mcp1', 'bh_pip1', 'bh_dip1', 'bh_act1'),
        ('index_bend_3',  'bh_mcp2', 'bh_pip2', 'bh_dip2', 'bh_act2'),
        ('middle_bend_3', 'bh_mcp3', 'bh_pip3', 'bh_dip3', 'bh_act3'),
        ('ring_bend_3',   'bh_mcp4', 'bh_pip4', 'bh_dip4', 'bh_act4'),
    ]
    fe_src_qpos = []      # DOGlove bend_3 qpos index
    fe_mcp_qpos = []      # bluehand mcp qpos index
    fe_pip_qpos = []      # bluehand pip qpos index
    fe_dip_qpos = []      # bluehand dip qpos index
    fe_act_qpos = []      # bluehand act qpos index
    for src, mcp, pip, dip, act in fe_mapping:
        fe_src_qpos.append(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, src)])
        fe_mcp_qpos.append(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mcp)])
        fe_pip_qpos.append(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pip)])
        fe_dip_qpos.append(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dip)])
        fe_act_qpos.append(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, act)])

    # bluehand 모든 joint (물리에서 보호)
    bh_all_joint_names = [
        "bh_aa1", "bh_mcp1", "bh_pip1", "bh_dip1", "bh_act1",
        "bh_aa2", "bh_mcp2", "bh_pip2", "bh_dip2", "bh_act2",
        "bh_aa3", "bh_mcp3", "bh_pip3", "bh_dip3", "bh_act3",
        "bh_aa4", "bh_mcp4", "bh_pip4", "bh_dip4", "bh_act4",
    ]
    bh_all_qpos_indices = []
    bh_all_qvel_indices = []
    for name in bh_all_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        bh_all_qpos_indices.append(model.jnt_qposadr[jid])
        bh_all_qvel_indices.append(model.jnt_dofadr[jid])

    try:
        with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            mujoco.mj_forward(model, data)

            rate = RateLimiter(frequency=100.0)

            while viewer.is_running():
                # 1) DOGlove FK: 글러브 데이터로 actuator 제어
                joints = receiver.get_most_recent_joints()
                if joints is not None:
                    for i in range(len(joints)):
                        data.ctrl[doglove_joint_ids[i]] = joints[i]

                # bluehand qpos 저장 (mj_step 물리 시뮬로부터 보호)
                saved_bh_qpos = np.array(
                    [data.qpos[idx] for idx in bh_all_qpos_indices]
                )

                mujoco.mj_step(model, data)

                # bluehand qpos 복원 + qvel 제로 (물리 시뮬 결과 완전 무시)
                for idx, val in zip(bh_all_qpos_indices, saved_bh_qpos):
                    data.qpos[idx] = val
                for idx in bh_all_qvel_indices:
                    data.qvel[idx] = 0.0

                # 2) AA 직접 매핑: DOGlove split → bluehand aa (엄지만 부호 반전 + 스케일)
                THUMB_AA_SCALE = 5.0
                for i, (src_idx, dst_idx) in enumerate(zip(aa_src_qpos, aa_dst_qpos)):
                    if i == 0:  # thumb
                        data.qpos[dst_idx] = -data.qpos[src_idx] * THUMB_AA_SCALE
                    else:
                        data.qpos[dst_idx] = data.qpos[src_idx]

                # 3) FE 매핑: DOGlove bend_3 → bluehand mcp, coupling → pip/dip/act
                #    DOGlove bend는 음수(굽힘), bluehand mcp는 양수(굽힘) → 부호 반전
                #    clamp >= 0 (역방향 꺾임 방지), scale 1.5 (굽힘 보정)
                FE_SCALE = 2.0
                for i in range(len(fe_src_qpos)):
                    mcp_val = max(0.0, -data.qpos[fe_src_qpos[i]] * FE_SCALE)
                    data.qpos[fe_mcp_qpos[i]] = mcp_val
                    data.qpos[fe_pip_qpos[i]] = coupling_pip(mcp_val)
                    data.qpos[fe_dip_qpos[i]] = coupling_dip(mcp_val)
                    data.qpos[fe_act_qpos[i]] = mcp_val * 0.8

                mujoco.mj_forward(model, data)

                # 4) ROS2 publish: 정규화된 AA/FE 값
                #    AA: radian → -1~1 (÷ π/2), FE: radian → 0~1 (÷ π/2)
                aa_vals = []
                fe_vals = []
                for i in range(4):  # finger1(thumb), 2(index), 3(middle), 4(ring)
                    aa_raw = -data.qpos[aa_dst_qpos[i]]
                    aa_vals.append(float(np.clip(aa_raw / 1.5708, -1.0, 1.0)))
                    fe_raw = data.qpos[fe_mcp_qpos[i]]
                    fe_vals.append(float(np.clip(fe_raw / 1.5708, 0.0, 1.0)))

                goal_msg = JointState()
                goal_msg.header.stamp = ros_node.get_clock().now().to_msg()
                goal_msg.name = [
                    'finger1_AA', 'finger1_FE',
                    'finger2_AA', 'finger2_FE',
                    'finger3_AA', 'finger3_FE',
                    'finger4_AA', 'finger4_FE',
                ]
                goal_msg.position = [
                    aa_vals[0], fe_vals[0],
                    aa_vals[1], fe_vals[1],
                    aa_vals[2], fe_vals[2],
                    aa_vals[3], fe_vals[3],
                ]
                goal_pub.publish(goal_msg)

                viewer.sync()
                rate.sleep()

    except KeyboardInterrupt:
        print("Program interrupted by user")

    finally:
        receiver.stop()
        ros_node.destroy_node()
        rclpy.shutdown()
        print("UDP receiver stopped successfully")

if __name__ == '__main__':
    main()
