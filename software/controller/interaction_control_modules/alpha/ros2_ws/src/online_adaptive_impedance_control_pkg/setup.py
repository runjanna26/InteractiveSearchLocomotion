from setuptools import setup

package_name = 'online_adaptive_impedance_control_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='runj',
    maintainer_email='run.janna@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'online_adaptive_impedance_control_node = online_adaptive_impedance_control_pkg.online_adaptive_impedance_control_node:main',
            'muscle_learning_node = online_adaptive_impedance_control_pkg.muscle_learning:main',
        ],
    },
)
