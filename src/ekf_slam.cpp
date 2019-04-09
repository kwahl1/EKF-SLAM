#define EIGEN_NO_AUTOMATIC_RESIZING = 1;

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
#include <iostream>

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
  VectorXf mu; // state [x,y,theta]
  MatrixXf sigma;
  Matrix3f R;
  Matrix2f Q;
  Vector3f previous_odom;
  Vector3f odom;
  float mahalanobis_threshold, std_motion_xy, std_motion_theta, std_sensor_range, std_sensor_bearing, std_new_landmark, uncertainty_scale, frequency;
  int N_landmarks;
  float sum_time;
  int N_times;

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
    mu = VectorXf::Zero(3,1);
    sigma = MatrixXf::Zero(3,3);
    odom = Vector3f::Zero();
    previous_odom = Vector3f::Zero();
    N_landmarks = 0;
    N_times = 0;

    // parameters loaded from config/params.yaml
    nh.param<float>("std_motion_xy", std_motion_xy, 0.04f);
    nh.param<float>("std_motion_theta", std_motion_theta, 0.09f);
    nh.param<float>("std_sensor_range", std_sensor_range, 0.005f);
    nh.param<float>("std_sensor_bearing", std_sensor_bearing, 0.5f);
    nh.param<float>("mahalanobis_threshold", mahalanobis_threshold, 0.98f);
    nh.param<float>("std_new_landmark", std_new_landmark, 0.1f);
    nh.param<float>("uncertainty_scale", uncertainty_scale, 0.1f);
    nh.param<float>("frequency", frequency, 10.0f);

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
    R = Matrix3f::Identity(3,3);
    R(0,0) = std_motion_xy;
    R(1,1) = std_motion_xy;
    R(2,2) = 2*M_PI*(std_motion_theta/360);

    //observation noise
    Q = Matrix2f::Identity(2,2);
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

    Vector3f g = odometryMotionModel();
    Matrix3f G = jacobianOdometryMotionModel();
    previous_odom = odom; // save for next iteration

    mu.head(3) = mu.head(3) + g;
    mu(2) = constrainAngle(mu(2));
    sigma.block(0,0,3,3) = G*sigma.block(0,0,3,3)*G.transpose()+R;

    assert(G.hasNaN() == false);
    assert(sigma.hasNaN() == false);
    assert(mu.hasNaN() == false);
    assert(odom.hasNaN() == false);

    ROS_INFO("Prediction done mu=[%f,%f,%f] sigma=[%f,%f,%f]",mu(0),mu(1),mu(2),sigma(0,0),sigma(1,1),sigma(2,2));
    return true;
  }


  /*
  Set heading angle to interval [PI, -PI]
  */
  float constrainAngle(float theta)
  {
    theta = std::fmod(theta + M_PI,2*M_PI);
    if (theta < 0)
    {
      theta += 2*M_PI;
    }
    return theta - M_PI;
  }


  /*
  Compute motion from odometry increment
  */
  Vector3f odometryMotionModel()
  {
    Vector3f g;
    float d_trans,d_rot1,d_rot2;

    d_trans = std::sqrt(pow(odom(0)-previous_odom(0),2)+pow(odom(1)-previous_odom(1),2));
    d_rot1 = std::atan2(odom(1)-previous_odom(1),odom(0)-previous_odom(0));
    d_rot2 = odom(2)-previous_odom(2)-d_rot1;

    g << d_trans*std::cos(mu(2)+d_rot1), d_trans*std::sin(mu(2)+d_rot1),d_rot1+d_rot2;

    return g;
  }


  /*
  Compute jacobian matrix of the odometry motion model
  */
  Matrix3f jacobianOdometryMotionModel()
  {
    Matrix3f G;
    float d_trans,d_rot1;

    d_trans = std::sqrt(pow(odom(0)-previous_odom(0),2)+pow(odom(1)-previous_odom(1),2));
    d_rot1 = std::atan2(odom(1)-previous_odom(1),odom(0)-previous_odom(0));

    G << 1, 0, -d_trans*std::sin(mu(2)+d_rot1),
    0, 1, d_trans*std::cos(mu(2)+d_rot1),
    0, 0, 1;

    return G;
  }


  /*
  Range bearing observation model.
  Computes the expected measurement range and bearing given robots current position and landmark position x,y
  */
  Vector2f observationModel(float x, float y)
  {
    Vector2f delta_k(x-mu(0),y-mu(1));
    float q_k = delta_k.transpose()*delta_k;
    float bearing = std::atan2(delta_k(1), delta_k(0)) - mu(2);

    return Vector2f(sqrt(q_k), constrainAngle(bearing));
  }


  /*
  Inverse range bearing observation model.
  Computes the landmark position x,y given the robots position and an observed range bearing
  */
  Vector2f inverseObservationModel(float range, float bearing)
  {
    return Vector2f(mu(0) + range*std::cos(mu(2)+bearing), mu(1) + range*std::sin(mu(2)+bearing));
  }


  /*
  Compute jacobian matrix of the observation model
  */
  MatrixXf jacobianObservationModel(float x, float y)
  {
    MatrixXf h(2,5);
    Vector2f delta_k(x-mu(0),y-mu(1));
    float q_k = delta_k.transpose()*delta_k;

    h << -std::sqrt(q_k)*delta_k(0), -std::sqrt(q_k)*delta_k(1), 0, std::sqrt(q_k)*delta_k(0), std::sqrt(q_k)*delta_k(1),
    delta_k(1), -delta_k(0), -1, -delta_k(1), delta_k(0);

    return (1/q_k)*h;
  }


/*
Add potentially new landmark to mu and sigma
*/
  void addNewLandmark(Vector2f z)
  {
    // append potentially new landmark to mu and sigma
    mu.conservativeResize(mu.rows()+2);
    mu.tail(2) = inverseObservationModel(z(0),z(1));

    sigma.conservativeResize(sigma.rows()+2,sigma.cols()+2);
    //sigma.block(0,3+2*N_landmarks,3+2*N_landmarks,2).setZero();
    sigma.block(0,3+2*N_landmarks,3+2*N_landmarks,2).setZero();
    sigma.block(3+2*N_landmarks,0,2,3+2*N_landmarks).setZero();
    sigma.bottomRightCorner(2,2) = std_new_landmark*Matrix2f::Identity();
    assert(sigma.hasNaN() == false);
  }


/*
Remove last landmark from mu and sigma
*/
  void removeNewLandmark()
  {
    mu.conservativeResize(mu.rows()-2);
    sigma.conservativeResize(sigma.rows()-2,sigma.cols()-2);
  }


/*
Perform maximum likelihood data association given observed landmarks.
*/
  void MLDataAssociation(const mrpt_msgs::ObservationRangeBearing::ConstPtr& msg)
  {
    int N_observations = msg->sensed_data.size();
    MatrixXf F_x_k, H, psi, nu;
    Matrix2f psi_k;
    VectorXf likelihood;
    Vector2f z_bar_k, nu_k;

    for(int i = 0; i < N_observations; i++)
    {
      // store results for data association
      likelihood.resize(N_landmarks+1);
      H.resize(2*(3+2*(1+N_landmarks)),N_landmarks+1);
      psi.resize(4,N_landmarks+1);
      nu.resize(2,N_landmarks+1);

      Vector2f z_i((float) msg->sensed_data[i].range, (float) msg->sensed_data[i].yaw);

      // append z_i as potentially new landmark to mu and sigma
      addNewLandmark(z_i);

      for (int k = 0; k < N_landmarks+1; k++)
      {
        z_bar_k = observationModel(mu(3+2*k),mu(4+2*k));

        F_x_k.resize(5,3+2*(N_landmarks+1));
        F_x_k.setZero();
        F_x_k(0,0) = 1;
        F_x_k(1,1) = 1;
        F_x_k(2,2) = 1;
        F_x_k(3,3+2*k) = 1;
        F_x_k(4,4+2*k) = 1;

        MatrixXf H_k;
        H_k.noalias() = jacobianObservationModel(mu(3+2*k),mu(4+2*k))*F_x_k;
        psi_k.noalias() = H_k*sigma*H_k.transpose()+Q;

        nu_k = z_i-z_bar_k;
        nu_k(1) = constrainAngle(nu_k(1));
        likelihood(k) = (1/(2*M_PI*sqrt(psi_k.determinant())))*exp(-0.5*nu_k.transpose()*psi_k.inverse()*nu_k);

        // save for later
        nu.col(k) = nu_k;
        H.col(k) = VectorXf(Map<VectorXf>(H_k.data(), H_k.cols()*H_k.rows()));
        psi.col(k) = Vector4f(Map<Vector4f>(psi_k.data(), psi_k.cols()*psi_k.rows()));

        //debug
        //assert(psi_k.hasNaN() == false);
        //assert(nu.hasNaN() == false);
        //assert(h_k.hasNaN() == false);
        //assert(H.hasNaN() == false);
        //assert(likelihood.hasNaN() == false);
      }
      likelihood(N_landmarks) = mahalanobis_threshold; // likelihood for new landmark
      VectorXf::Index ind;
      likelihood.maxCoeff(&ind); // get index for maximum likelihood
      int j_i = (int) ind;

      MatrixXf H_k;
      if (j_i == N_landmarks)
      {
        //observation is a new landmark
        N_landmarks++;
      }else
      {
        // not a new landmark
        removeNewLandmark(); // remove potentially new landmark from mu_bar and sigma_bar
      }
      H_k.resize(2,3+2*N_landmarks);
      H_k = Map<MatrixXf>( H.col(j_i).data(),2,3+2*N_landmarks);

      MatrixXf K;
      psi_k =  Map<Matrix2f>( psi.col(j_i).data(),2,2);
      K.noalias() = sigma*H_k.transpose()*psi_k.inverse();

      // Update state belief
      mu.noalias() += K*nu.col(j_i);
      sigma = (MatrixXf::Identity(sigma.rows(),sigma.cols())-K*H_k)*sigma;
      mu(2) = constrainAngle(mu(2));
    }
  }


  /*
  Callback function for landmark observations.
  */
  void landmarkCallback(const mrpt_msgs::ObservationRangeBearing::ConstPtr& msg)
  {
    timeLastMsg = msg->header.stamp;

    if(!predictMotion())
    {
      ROS_INFO("Skipped iteration.");
      return;
    }
    ROS_INFO("Prediction done");

    start_ = ros::WallTime::now();
    MLDataAssociation(msg);
    ROS_INFO("Update done.\n mu(%f,%f,%f) \nsigma (%f,%f,%f)\nN_landmarks = %i",mu(0),mu(1),mu(2),sigma(0,0),sigma(1,1),sigma(2,2),N_landmarks);
    end_ = ros::WallTime::now();
    float timex = (end_ - start_).toNSec() * 1e-6;
    ROS_INFO("Update exectution time (ms): %f",timex);
    sum_time += timex;
    N_times++;
    ROS_INFO("avg = %f",sum_time/N_times);

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
  sleep(0.5);
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
