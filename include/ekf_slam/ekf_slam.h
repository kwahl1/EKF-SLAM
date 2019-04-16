#define EIGEN_NO_AUTOMATIC_RESIZING = 1;

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <sensor_msgs/PointCloud.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <mrpt_msgs/ObservationRangeBearing.h>
#include <ros/ros.h>
#include <cmath>
#include <math.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <cassert>
#include <iostream>



