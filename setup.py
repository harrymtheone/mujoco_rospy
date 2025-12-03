from setuptools import setup
import os
from glob import glob

package_name = 'mujoco_rospy'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files.
        # (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Python wrapper for MuJoCo with ROS 2 interface',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mujoco_node = mujoco_rospy.mujoco_node:main',
        ],
    },
)

