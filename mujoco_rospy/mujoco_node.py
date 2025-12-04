import time
import threading
import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty, SetBool
from tf2_ros import TransformBroadcaster
from mujoco_ros_msgs.msg import JointControlCmd

class MujocoRosNode(Node):
    def __init__(self):
        super().__init__('mujoco_node')

        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('publish_rate', 500.0)
        self.declare_parameter('headless', False)
        
        self.model_path = self.get_parameter('model_path').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.headless = self.get_parameter('headless').value

        if not self.model_path:
            self.get_logger().error('model_path parameter is required')
            raise ValueError('model_path parameter is required')

        # Load Model
        self.get_logger().info(f'Loading model from {self.model_path}')
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Sim State
        self.sim_running = True
        self.paused = False
        self.reset_request = False
        
        # Cache IDs
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                           for i in range(self.model.njnt)]
        # Filter out None names (unnamed joints)
        self.joint_names = [n for n in self.joint_names if n]
        
        self.actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) 
                              for i in range(self.model.nu)]
        
        self._build_mappings()

        # Command buffers
        self.cmd_mutex = threading.Lock()
        self.target_pos = np.zeros(self.model.nu)
        self.target_vel = np.zeros(self.model.nu)
        self.target_tau = np.zeros(self.model.nu)
        self.kp = np.zeros(self.model.nu)
        self.kd = np.zeros(self.model.nu)

        # Publishers
        self.pub_joint_state = self.create_publisher(JointState, '/mujoco/joint_states', 10)
        self.pub_imu = self.create_publisher(Imu, '/mujoco/imu', 10)
        self.pub_odom = self.create_publisher(Odometry, '/mujoco/odom', 10)
        
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscriber
        self.sub_cmd = self.create_subscription(
            JointControlCmd, '/mujoco/joint_cmd', self._cmd_callback, 10)

        # Services
        self.srv_reset = self.create_service(Empty, '/mujoco/reset', self._reset_callback)
        self.srv_pause = self.create_service(SetBool, '/mujoco/pause', self._pause_callback)

        # Start simulation thread
        self.sim_thread = threading.Thread(target=self._sim_loop)
        self.sim_thread.start()

    def _build_mappings(self):
        # Map joint name to qpos/qvel address
        self.jnt_qposadr = {}
        self.jnt_dofadr = {}
        
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.jnt_qposadr[name] = self.model.jnt_qposadr[i]
                self.jnt_dofadr[name] = self.model.jnt_dofadr[i]
        
        # Map actuator name to id
        self.actuator_map = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.actuator_map[name] = i
            else:
                # Try to find by joint name if actuator is unnamed but attached to joint
                # This is a simplification; robust logic would check trnid
                pass

    def _cmd_callback(self, msg):
        with self.cmd_mutex:
            if len(msg.position) > 0:
                self.target_pos[:len(msg.position)] = msg.position
            if len(msg.velocity) > 0:
                self.target_vel[:len(msg.velocity)] = msg.velocity
            if len(msg.torque) > 0:
                self.target_tau[:len(msg.torque)] = msg.torque
            if len(msg.kp) > 0:
                self.kp[:len(msg.kp)] = msg.kp
            if len(msg.kd) > 0:
                self.kd[:len(msg.kd)] = msg.kd

    def _reset_callback(self, request, response):
        self.reset_request = True
        return response

    def _pause_callback(self, request, response):
        self.paused = request.data
        response.success = True
        response.message = "Paused" if self.paused else "Resumed"
        return response

    def _apply_control(self):
        with self.cmd_mutex:
            # PD Control calculation
            # This assumes 1-1 mapping for actuators usually
            # For this simplified node, we iterate actuators
            
            # We need to map actuators to qpos/qvel
            # model.actuator_trnid[i, 0] gives the joint ID for joint-based actuators
            
            ctrl = np.zeros(self.model.nu)
            
            for i in range(self.model.nu):
                # Setup targets
                q_des = self.target_pos[i]
                dq_des = self.target_vel[i]
                tau_ff = self.target_tau[i]
                kp = self.kp[i]
                kd = self.kd[i]
                
                # Get current state
                # Find which joint this actuator acts on
                # trntype 0 = joint
                if self.model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                    joint_id = self.model.actuator_trnid[i, 0]
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    dof_adr = self.model.jnt_dofadr[joint_id]
                    
                    q_cur = self.data.qpos[qpos_adr]
                    dq_cur = self.data.qvel[dof_adr]
                    
                    # Compute torque
                    tau = kp * (q_des - q_cur) + kd * (dq_des - dq_cur) + tau_ff
                    ctrl[i] = tau
                else:
                    # Fallback for other actuator types (e.g. just feedforward)
                    ctrl[i] = tau_ff
            
            self.data.ctrl[:] = ctrl

    def _publish_state(self):
        now = self.get_clock().now().to_msg()
        
        # 1. Joint States
        js = JointState()
        js.header.stamp = now
        js.name = self.joint_names
        js.position = []
        js.velocity = []
        
        for name in self.joint_names:
            js.position.append(self.data.qpos[self.jnt_qposadr[name]])
            js.velocity.append(self.data.qvel[self.jnt_dofadr[name]])
            # Effort is trickier, keeping simple for now
            js.effort.append(0.0)
            
        self.pub_joint_state.publish(js)
        
        # 2. IMU
        # Assumes site names "imu_site" or sensors "imu_quat", "imu_gyro", "imu_accel"
        
        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = "base_link"
        
        # Orientation (Quaternion)
        # Look for 'imu_quat' sensor
        id_quat = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
        id_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu_site")
        
        if id_quat >= 0 and id_site >= 0:
            adr = self.model.sensor_adr[id_quat]
            imu.orientation.w = self.data.sensordata[adr]
            imu.orientation.x = self.data.sensordata[adr+1]
            imu.orientation.y = self.data.sensordata[adr+2]
            imu.orientation.z = self.data.sensordata[adr+3]
        elif id_site >= 0:
            # Fallback to site orientation if "imu_site" exists
            mat = self.data.site_xmat[id_site].reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, mat.flatten())
            imu.orientation.w = quat[0]
            imu.orientation.x = quat[1]
            imu.orientation.y = quat[2]
            imu.orientation.z = quat[3]
        else:
            raise ValueError("IMU Orientation sensor 'imu_quat' or site 'imu_site' not found in model")

        # Gyro
        id_gyro = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        if id_gyro >= 0:
            adr = self.model.sensor_adr[id_gyro]
            imu.angular_velocity.x = self.data.sensordata[adr]
            imu.angular_velocity.y = self.data.sensordata[adr+1]
            imu.angular_velocity.z = self.data.sensordata[adr+2]
        else:
            raise ValueError("IMU Gyro sensor 'imu_gyro' not found in model")

        # Accel
        id_accel = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        if id_accel >= 0:
            adr = self.model.sensor_adr[id_accel]
            imu.linear_acceleration.x = self.data.sensordata[adr]
            imu.linear_acceleration.y = self.data.sensordata[adr+1]
            imu.linear_acceleration.z = self.data.sensordata[adr+2]
                
        self.pub_imu.publish(imu)
        
        # 3. Odometry
        # Using 'imu_site' for odometry position
        id_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu_site")
        
        if id_site >= 0:
            odom = Odometry()
            odom.header.stamp = now
            odom.header.frame_id = "odom"
            odom.child_frame_id = "base_link"
            
            # Position from imu_site
            odom.pose.pose.position.x = self.data.site_xpos[id_site][0]
            odom.pose.pose.position.y = self.data.site_xpos[id_site][1]
            odom.pose.pose.position.z = self.data.site_xpos[id_site][2]
            
            # Orientation from imu_site (site_xmat to quat)
            mat = self.data.site_xmat[id_site].reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, mat.flatten())
            
            odom.pose.pose.orientation.w = quat[0]
            odom.pose.pose.orientation.x = quat[1]
            odom.pose.pose.orientation.y = quat[2]
            odom.pose.pose.orientation.z = quat[3]
            
            # Twist (requires qvel mapping to body velocity - simplified here)
            # mj_objectVelocity would be better but for free joint root:
            odom.twist.twist.linear.x = self.data.qvel[0]
            odom.twist.twist.linear.y = self.data.qvel[1]
            odom.twist.twist.linear.z = self.data.qvel[2]
            odom.twist.twist.angular.x = self.data.qvel[3]
            odom.twist.twist.angular.y = self.data.qvel[4]
            odom.twist.twist.angular.z = self.data.qvel[5]
            
            self.pub_odom.publish(odom)


    def _sim_loop(self):
        
        # Physics step
        def physics_step(model, data):
            if self.reset_request:
                mujoco.mj_resetData(model, data)
                self.reset_request = False
                
            if not self.paused:
                self._apply_control()
                mujoco.mj_step(model, data)

        # Viewer wrapper
        if not self.headless:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self._run_loop(viewer, physics_step)
        else:
            self._run_loop(None, physics_step)

    def _run_loop(self, viewer, physics_step):
        target_dt = self.model.opt.timestep
        last_pub_time = 0
        last_render_time = 0
        render_dt = 1.0 / 30.0  # Cap rendering at 30 FPS
        
        while self.sim_running:
            step_start = time.time()

            # Step physics
            physics_step(self.model, self.data)
            
            # Sync viewer if needed (throttled)
            if viewer and viewer.is_running():
                if time.time() - last_render_time >= render_dt:
                    viewer.sync()
                    last_render_time = time.time()
            
            # Publish state (throttled)
            # Using simulation time or wall time? Simulation time is better for ROS
            # But for now we use wall clock throttling
            if time.time() - last_pub_time >= (1.0 / self.publish_rate):
                self._publish_state()
                last_pub_time = time.time()
            
            # Sleep to maintain real-time factor
            elapsed = time.time() - step_start
            if elapsed < target_dt:
                 time.sleep(target_dt - elapsed)

def main(args=None):
    rclpy.init(args=args)
    node = MujocoRosNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sim_running = False
        node.sim_thread.join()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
