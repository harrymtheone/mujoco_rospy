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
        self.declare_parameter('render_rate', 60.0)  # Default to 60 FPS instead of 30
        
        self.model_path = self.get_parameter('model_path').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.headless = self.get_parameter('headless').value
        self.render_rate = self.get_parameter('render_rate').value

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

    def _quat_rotate(self, quat, vec):
        """Rotate a vector by a quaternion: q * v * q^-1
        
        Args:
            quat: Quaternion [w, x, y, z]
            vec: Vector [x, y, z]
        
        Returns:
            Rotated vector [x, y, z]
        """
        # Extract quaternion components
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        vx, vy, vz = vec[0], vec[1], vec[2]
        
        # Quaternion-vector multiplication (optimized formula)
        # t = 2 * cross(q.xyz, v)
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        
        # result = v + w * t + cross(q.xyz, t)
        return np.array([
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx)
        ])

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
        
        # Check imu_site exists first (required for odometry too)
        id_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu_site")
        if id_site < 0:
            raise ValueError("Site 'imu_site' not found in model")
        
        # Orientation (Quaternion)
        id_quat = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
    
        if id_quat >= 0:
            adr = self.model.sensor_adr[id_quat]
            imu.orientation.w = self.data.sensordata[adr]
            imu.orientation.x = self.data.sensordata[adr+1]
            imu.orientation.y = self.data.sensordata[adr+2]
            imu.orientation.z = self.data.sensordata[adr+3]
        else:
            raise ValueError("IMU Quat sensor 'imu_quat' not found in model")

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
        else:
            raise ValueError("IMU Accel sensor 'imu_accel' not found in model")
            
        self.pub_imu.publish(imu)
        
        # 3. Odometry (id_site already checked above)
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
            
        # Transform world-frame velocity to body-frame
        # qvel[0:3] is linear velocity in WORLD frame
        # qvel[3:6] is angular velocity in WORLD frame
        lin_vel_world = np.array([self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]])
        ang_vel_world = np.array([self.data.qvel[3], self.data.qvel[4], self.data.qvel[5]])
        
        # Rotate world velocity to body frame using quaternion inverse
        # quat is [w, x, y, z], need to rotate by conjugate (inverse)
        quat_conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
        lin_vel_body = self._quat_rotate(quat_conj, lin_vel_world)
        ang_vel_body = self._quat_rotate(quat_conj, ang_vel_world)
        
        odom.twist.twist.linear.x = lin_vel_body[0]
        odom.twist.twist.linear.y = lin_vel_body[1]
        odom.twist.twist.linear.z = lin_vel_body[2]
        odom.twist.twist.angular.x = ang_vel_body[0]
        odom.twist.twist.angular.y = ang_vel_body[1]
        odom.twist.twist.angular.z = ang_vel_body[2]
            
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
        render_dt = 1.0 / self.render_rate if self.render_rate > 0 else 0
        next_render_time = time.time()  # Target time for next render

        while self.sim_running:
            step_start = time.time()

            # Step physics
            physics_step(self.model, self.data)
            
            # Sync viewer if needed (throttled or unlimited)
            if viewer and viewer.is_running():
                current_time = time.time()
                should_render = (self.render_rate <= 0) or (current_time >= next_render_time)
                if should_render:
                    viewer.sync()

                    # Schedule next render at fixed interval from target (not from now)
                    # This maintains accurate frame rate regardless of sync() duration
                    if self.render_rate > 0:
                        next_render_time += render_dt
                        # If we've fallen behind (missed frames), catch up to current time
                        if next_render_time < current_time:
                            next_render_time = current_time + render_dt
            
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
