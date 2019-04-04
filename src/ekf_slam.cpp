#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <sensor_msgs/PointCloud.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <mrpt_msgs/ObservationRangeBearing.h>
#include <ros/ros.h>
#include <cmath>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <cassert>


using namespace Eigen;

class EKFslam
{
public:
  ros::NodeHandle nh;
  tf::TransformListener tf_listener;
  tf::TransformBroadcaster tf_broadcaster;
  ros::Subscriber sub_landmarks;
  ros::Publisher pub_mu;
  ros::Publisher pub_pcl;
  ros::Publisher pub_markers;
  ros::Time timeLastMsg;
  ros::WallTime start_, end_;
  VectorXd mu; // state [x,y,theta]
  MatrixXd sigma;
  Matrix3d R;
  Matrix2d Q;
  Vector3d previous_odom;
  Vector3d odom;
  double mahalanobis_threshold, std_motion_xy, std_motion_theta, std_sensor_range, std_sensor_bearing, std_new_landmark, uncertainty_scale, frequency;
  int N_landmarks;

  EKFslam(ros::NodeHandle n)
  {
    nh = n;
    sub_landmarks = nh.subscribe<mrpt_msgs::ObservationRangeBearing>("/landmark",1,&EKFslam::landmarkCallback,this);
    pub_mu = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/ekf_state",1);
    pub_pcl= nh.advertise<sensor_msgs::PointCloud>("/ekf_map",1);
    pub_markers = nh.advertise<visualization_msgs::MarkerArray>( "/ekf_covariance",1);
    this->init();
  }

  void init()
  {
    mu = VectorXd::Zero(3,1);
    sigma = MatrixXd::Zero(3,3);
    odom = Vector3d::Zero();
    previous_odom = Vector3d::Zero();
    N_landmarks = 0;

    // parameters loaded from config/params.yaml
    nh.param<double>("std_motion_xy", std_motion_xy, 0.04);
    nh.param<double>("std_motion_theta", std_motion_theta, 0.09);
    nh.param<double>("std_sensor_range", std_sensor_range, 0.005);
    nh.param<double>("std_sensor_bearing", std_sensor_bearing, 0.5);
    nh.param<double>("mahalanobis_threshold", mahalanobis_threshold, 0.98);
    nh.param<double>("std_new_landmark", std_new_landmark, 0.1);
    nh.param<double>("uncertainty_scale", uncertainty_scale, 0.1);
    nh.param<double>("frequency", frequency, 10.0);

    ROS_INFO("--- EKF SLAM Parameters ---");
    ROS_INFO("frequency: %f", frequency);
    ROS_INFO("std_motion_xy: %f", std_motion_xy);
    ROS_INFO("std_motion_theta: %f", std_motion_theta);
    ROS_INFO("std_sensor_range: %f", std_sensor_range);
    ROS_INFO("std_sensor_bearing: %f", std_sensor_bearing);
    ROS_INFO("mahalanobis_threshold: %f", mahalanobis_threshold);
    ROS_INFO("std_new_landmark: %f", std_new_landmark);
    ROS_INFO("---------------------------");

    //motion noise
    R = Matrix3d::Identity(3,3);
    R(0,0) = std_motion_xy;
    R(1,1) = std_motion_xy;
    R(2,2) = 2*M_PI*(std_motion_theta/360);

    //observation noise
    Q = Matrix2d::Identity(2,2);
    Q(0,0) = std_sensor_range;
    Q(1,1) = 2*M_PI*(std_sensor_bearing/360);
  }

  /*
  Get robot pose from odometry at time = timeLastMsg
  */
  bool getOdometryFromTF()
  {
    tf::StampedTransform transform;
    try
    {
      tf_listener.waitForTransform("/odom", "/base_link",
      timeLastMsg, ros::Duration(0.2));
      tf_listener.lookupTransform("/odom", "/base_link", timeLastMsg, transform);
    }
    catch (tf::TransformException ex){
      ROS_WARN("EKF_SLAM: %s",ex.what());
      return false;
    }

    odom(0) = transform.getOrigin().x();
    odom(1) = transform.getOrigin().y();
    odom(2) = tf::getYaw(transform.getRotation());
    odom(2) = constrainAngle(odom(2));

    return true;
  }


  /*
  EKF SLAM prediction step
  */
  bool predictMotion()
  {
    if(!getOdometryFromTF())
    {
      //no transform available. Skip iteration.
      return false;
    }

    Vector3d g = odometryMotionModel();
    Matrix3d G = jacobianOdometryMotionModel();
    previous_odom = odom; // save for next iteration

    mu.head(3) = mu.head(3) + g;
    mu(2) = constrainAngle(mu(2));
    sigma.block(0,0,3,3) = G*sigma.block(0,0,3,3)*G.transpose()+R;

    assert(G.hasNaN() == false);
    assert(sigma.hasNaN() == false);

    ROS_INFO("Prediction done mu=[%f,%f,%f] sigma=[%f,%f,%f]",mu(0),mu(1),mu(2),sigma(0,0),sigma(1,1),sigma(2,2));
    return true;
  }


  /*
  Set heading angle to interval [PI, -PI]
  */
  double constrainAngle(double theta)
  {
    theta = std::fmod((double) theta + M_PI, (double) 2*M_PI);
    if (theta < 0)
    {
      theta += 2*M_PI;
    }
    return theta - M_PI;
  }


  /*
  Compute motion from odometry increment
  */
  Vector3d odometryMotionModel()
  {
    Vector3d g;
    double d_trans,d_rot1,d_rot2;

    d_trans = std::sqrt(pow(odom(0)-previous_odom(0),2)+pow(odom(1)-previous_odom(1),2));
    d_rot1 = std::atan2(odom(1)-previous_odom(1),odom(0)-previous_odom(0));
    d_rot2 = odom(2)-previous_odom(2)-d_rot1;

    g << d_trans*cos(mu(2)+d_rot1), d_trans*sin(mu(2)+d_rot1),d_rot1+d_rot2;

    return g;
  }


  /*
  Compute jacobian matrix of the odometry motion model
  */
  Matrix3d jacobianOdometryMotionModel()
  {
    Matrix3d G;
    double d_trans,d_rot1;

    d_trans = std::sqrt(pow(odom(0)-previous_odom(0),2)+pow(odom(1)-previous_odom(1),2));
    d_rot1 = std::atan2(odom(1)-previous_odom(1),odom(0)-previous_odom(0));

    G << 1, 0, -d_trans*sin(mu(2)+d_rot1),
    0, 1, d_trans*cos(mu(2)+d_rot1),
    0, 0, 1;

    return G;
  }


  /*
  Range bearing observation model.
  Computes the expected measurement range and bearing given robots current position and landmark position x,y
  */
  Vector2d observationModel(double x, double y)
  {
    Vector2d delta_k(x-mu(0),y-mu(1));
    double q_k = delta_k.transpose()*delta_k;
    double bearing = std::atan2(delta_k(1), delta_k(0)) - mu(2);

    return Vector2d(sqrt(q_k), constrainAngle(bearing));
  }

  /*
  Inverse range bearing observation model.
  Computes the landmark position x,y given the robots position and an observed range bearing
  */
  Vector2d inverseObservationModel(double range, double bearing)
  {
    return Vector2d(mu(0) + range*cos(mu(2)+bearing), mu(1) + range*sin(mu(2)+bearing));
  }

  /*
  Compute jacobian matrix of the observation model
  */
  MatrixXd jacobianObservationModel(double x, double y)
  {
    MatrixXd h(2,5);
    Vector2d delta_k(x-mu(0),y-mu(1));
    double q_k = delta_k.transpose()*delta_k;

    h << -sqrt(q_k)*delta_k(0), -sqrt(q_k)*delta_k(1), 0, sqrt(q_k)*delta_k(0), sqrt(q_k)*delta_k(1),
    delta_k(1), -delta_k(0), -1, -delta_k(1), delta_k(0);

    return (1/q_k)*h;
  }


  /*
  Callback function for landmark observations. Performs maximum likelihood data association and EKF update step.

  TODO: Landmark loop to pure matrix operations.
  TODO: Move data association and update to separate functions.
  */
  void landmarkCallback(const mrpt_msgs::ObservationRangeBearing::ConstPtr& msg)
  {
    start_ = ros::WallTime::now();

    timeLastMsg = msg->header.stamp;
    int N_observations = msg->sensed_data.size();

    ROS_INFO("Got message. N_landmarks %i \n mu (%i) \n sigma %i,%i",N_landmarks,mu.size(),sigma.rows(),sigma.cols());
    if(!predictMotion())
    {
      ROS_INFO("Skipped iteration.");
      return;
    }
    ROS_INFO("Prediction done");

    for(int i = 0; i < N_observations; i++)
    {
      // store results for data association
      VectorXd likelihood = VectorXd::Zero(N_landmarks+1,1);
      MatrixXd H(2*(3+2*(1+N_landmarks)),N_landmarks+1);
      MatrixXd psi(4,N_landmarks+1);
      MatrixXd z_bar(2,N_landmarks+1);

      Vector2d z_i(msg->sensed_data[i].range,msg->sensed_data[i].yaw);

      // append potentially new landmark to mu and sigma
      mu.conservativeResize(mu.rows()+2);
      mu.tail(2) = inverseObservationModel(z_i(0),z_i(1));

      sigma.conservativeResize(sigma.rows()+2,sigma.cols()+2);
      sigma.block(0,3+2*N_landmarks,3+2*N_landmarks,2).setZero();
      sigma.block(3+2*N_landmarks,0,2,3+2*N_landmarks).setZero();
      sigma.bottomRightCorner(2,2) = std_new_landmark*Matrix2d::Identity();

      for (int k = 0; k < N_landmarks+1; k++)
      {

        Vector2d z_bar_k = observationModel(mu(3+2*k),mu(4+2*k));
        MatrixXd F_x_k(5,3+2*(N_landmarks+1));
        F_x_k << MatrixXd::Identity(3,3+2*(1+N_landmarks)),
        MatrixXd::Zero(2,3+2*(1+N_landmarks));
        F_x_k(3,3+2*k) = 1;
        F_x_k(4,4+2*k) = 1;

        MatrixXd h_k = jacobianObservationModel(mu(3+2*k),mu(4+2*k));
        MatrixXd H_k = h_k*F_x_k;
        Matrix2d psi_k = H_k*sigma*H_k.transpose()+Q;

        Vector2d nu = z_i-z_bar_k;
        nu(1) = constrainAngle(nu(1));
        likelihood(k) = (1/(2*M_PI*sqrt(psi_k.determinant())))*exp(-0.5*nu.transpose()*psi_k.inverse()*nu);

        // save for later
        z_bar.col(k) = z_bar_k;
        H.col(k) = VectorXd(Map<VectorXd>(H_k.data(), H_k.cols()*H_k.rows()));
        psi.col(k) = Vector4d(Map<Vector4d>(psi_k.data(), psi_k.cols()*psi_k.rows()));

        //debug
        assert(psi_k.hasNaN() == false);
        assert(nu.hasNaN() == false);
        assert(likelihood.hasNaN() == false);
      }
      likelihood(N_landmarks) = mahalanobis_threshold; // likelihood for new landmark
      VectorXd::Index ind;
      likelihood.maxCoeff(&ind); // get index for maximum likelihood
      int j_i = (int) ind;

      MatrixXd H_k;
      if (j_i+1 > N_landmarks)
      {
        //observation is a new landmark
        H_k = MatrixXd::Zero(2,3+2*(1+N_landmarks));
        H_k = Map<MatrixXd>( H.col(j_i).data(),2,3+2*(1+N_landmarks));

        N_landmarks++;
      }else
      {
        // not a new landmark
        H_k = MatrixXd::Zero(2,3+2*N_landmarks);
        H_k = Map<MatrixXd>( H.col(j_i).head(2*(3+2*N_landmarks)).data(),2,3+2*N_landmarks);

        // remove potentially new landmark from mu_bar and sigma_bar
        mu.conservativeResize(mu.rows()-2);
        sigma.conservativeResize(sigma.rows()-2,sigma.cols()-2);
      }

      Matrix2d psi_k;
      psi_k =  Map<Matrix2d>( psi.col(j_i).data(),2,2);
      MatrixXd K = sigma*H_k.transpose()*psi_k.inverse();

      // Update state belief
      mu = mu + K*(z_i-z_bar.col(j_i));
      MatrixXd I = MatrixXd::Identity(sigma.rows(),sigma.cols());
      sigma = (I-K*H_k)*sigma;
      mu(2) = constrainAngle(mu(2));

    }
    ROS_INFO("N_landmarks = %i",N_landmarks);
    ROS_INFO("Update done.\n mu(%f,%f,%f) \nsigma (%f,%f,%f)",mu(0),mu(1),mu(2),sigma(0,0),sigma(1,1),sigma(2,2));

    end_ = ros::WallTime::now();
    ROS_INFO("Exectution time (ms): %f",(end_ - start_).toNSec() * 1e-6);

    publishTF();
    publishPose();
    publishPointCloud();
    publishMarkers();
  }


  /*
  Publish transform from frame map->odom
  */
  void publishTF()
  {
    tf::Stamped<tf::Pose> odom_to_map;
    tf::Quaternion quat;
    tf::Transform temp;

    temp.setOrigin( tf::Vector3(mu(0), mu(1), 0.0) );
    quat.setRPY(0.0, 0.0, mu(2));
    temp.setRotation(quat);

    try
    {
      tf::Stamped<tf::Pose> temp_stamped(temp.inverse(), timeLastMsg, "/base_link");
      tf_listener.transformPose("/odom", temp_stamped, odom_to_map);
    }
    catch (tf::TransformException ex)
    {
      ROS_WARN("Failed to get transform from /odom to /base_link: %s",ex.what());
      return;
    }

    tf::Transform odom_to_map_tf = tf::Transform(tf::Quaternion(odom_to_map.getRotation()), tf::Point(odom_to_map.getOrigin()));
    tf::StampedTransform map_to_odom(odom_to_map_tf.inverse(), timeLastMsg, "/map", "/odom");
    tf_broadcaster.sendTransform(map_to_odom);
  }


  //---------------------------------- Visualization methods ----------------------------------

  /*
  Publish landmarks as visualization_msgs::Marker::CYLINDER with scale representing the uncertainty
  */
  void publishMarkers()
  {
    visualization_msgs::MarkerArray markerarray;

    for (int i = 0; i < N_landmarks; i++)
    {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "/map";
      marker.type = visualization_msgs::Marker::CYLINDER;
      marker.action = 0;
      marker.pose.position.x = mu(3+2*i);
      marker.pose.position.y = mu(4+2*i);
      if (sigma(3+2*i,3+2*i) == 0 || sigma(4+2*i,4+2*i) == 0)
      {
        ROS_WARN("Sigma for landmark is 0");
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 1;
      }else
      {
        marker.scale.x = uncertainty_scale*sigma(3+2*i,3+2*i);
        marker.scale.y = uncertainty_scale*sigma(4+2*i,4+2*i);
        marker.scale.z = 0.05;
      }
      marker.color.a = 1.0;
      marker.color.r = 1.0;
      marker.color.g = 1.0;
      marker.color.b = 1.0;
      marker.id = i;

      markerarray.markers.push_back(marker);
    }
    pub_markers.publish(markerarray);
  }


  /*
  Publish landmarks as pointcloud
  */
  void publishPointCloud()
  {
    sensor_msgs::PointCloud map;
    geometry_msgs::Point32 point;

    map.header.frame_id = "/map";
    for (int i = 0; i < N_landmarks; i++)
    {
      point.x = mu(3+2*i);
      point.y = mu(4+2*i);
      point.z = 0.0;
      map.points.push_back(point);
    }
    pub_pcl.publish(map);
  }


  /*
  Publish pose as geometry_msgs::PoseWithCovarianceStamped
  */
  void publishPose()
  {
    geometry_msgs::PoseWithCovarianceStamped pose_msg;
    geometry_msgs::Quaternion quat_msg;
    tf::Quaternion quat;

    pose_msg.header.frame_id = "/map";
    pose_msg.header.stamp = ros::Time::now(); //timeLastUpdate;

    pose_msg.pose.pose.position.x = mu(0);
    pose_msg.pose.pose.position.y = mu(1);
    pose_msg.pose.pose.position.z = 0.0;

    quat.setRPY(0.0, 0.0, mu(2));
    tf::quaternionTFToMsg(quat, quat_msg);
    pose_msg.pose.pose.orientation = quat_msg;

    pose_msg.pose.covariance[0] = sigma(0,0);
    pose_msg.pose.covariance[7] = sigma(1,1);
    pose_msg.pose.covariance[35] = sigma(2,2);

    pub_mu.publish(pose_msg);
  }
};



int main(int argc, char **argv)
{
  ros::init(argc, argv, "ekf_slam");
  ros::NodeHandle n("~");
  EKFslam slamNode(n);
  ros::Rate r(slamNode.frequency);

  while (ros::ok())
  {
    ros::spinOnce();
    r.sleep();
  }
  return 0;
}
