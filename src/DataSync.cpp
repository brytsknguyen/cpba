#include <thread>     // Include the <thread> header for sleep_for
#include <chrono>     // Include the <chrono> header for time durations

#include "ros/ros.h"
#include <rosbag/view.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

#include "nav_msgs/Odometry.h"
#include "sensor_msgs/PointCloud2.h"

#include "utility.h"

// #define KNRM  "\x1B[0m"
// #define KRED  "\x1B[31m"
// #define KGRN  "\x1B[32m"
// #define KYEL  "\x1B[33m"
// #define KBLU  "\x1B[34m"
// #define KMAG  "\x1B[35m"
// #define KCYN  "\x1B[36m"
// #define KWHT  "\x1B[37m"
// #define RESET "\033[0m"

using namespace std;

ros::NodeHandlePtr nh_ptr;

typedef nav_msgs::Odometry::Ptr odomMsg;
typedef sensor_msgs::PointCloud2::Ptr cloudMsg;

struct SyncedData
{
    odomMsg odom_data;
    odomMsg atag_data;
    cloudMsg cloud_data;
};

deque<odomMsg> atag_data;
deque<odomMsg> odom_data;
deque<cloudMsg> cloud_data;
deque<SyncedData> synchronized_data;


int main(int argc, char **argv)
{
    ros::init(argc, argv, "data_sync");

    ros::NodeHandle nh("~");
    nh_ptr = boost::make_shared<ros::NodeHandle>(nh);
    
    // printf("Hello world\n");

    // Get the bag file name
    string bag_file;
    nh_ptr->param("bag_file", bag_file, string(""));
    printf("Openning bag file: %s\n", bag_file.c_str());

    double tag_time_tolerance;
    nh_ptr->param("tag_time_tolerance", tag_time_tolerance, 0.05);
    
    rosbag::Bag bag;
    bag.open(bag_file, rosbag::bagmode::Read);

    // Query the topics
    vector<string> topics = {"/tag_pose", "/Odometry", "/cloud_registered"};
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    for (rosbag::MessageInstance & m : view)
    {
        if (m.getTopic() == "/tag_pose")
        {
            // Load the data into the tag buffer
            nav_msgs::Odometry::Ptr msg = m.instantiate<nav_msgs::Odometry>();
            msg->header.frame_id = "body";

            // Process the message data
            // ROS_INFO("Topic: %s. Time: %f", m.getTopic().c_str(), msg->header.stamp.toSec());

            // Store tag data
            atag_data.push_back(msg);
        }

        if (m.getTopic() == "/Odometry")
        {           
            // Load the data into the tag buffer
            nav_msgs::Odometry::Ptr msg = m.instantiate<nav_msgs::Odometry>();
            msg->header.frame_id = "world";
            msg->child_frame_id = "body";

            // Process the message data
            // ROS_INFO("Topic: %s. Time: %f", m.getTopic().c_str(), msg->header.stamp.toSec());

            // April tag data
            odom_data.push_back(msg);
        }

        if (m.getTopic() == "/cloud_registered")
        {           
            // Load the data into the tag buffer
            sensor_msgs::PointCloud2::Ptr msg = m.instantiate<sensor_msgs::PointCloud2>();
            msg->header.frame_id = "world";

            // Process the message data
            // ROS_INFO("Topic: %s. Time: %f", m.getTopic().c_str(), msg->header.stamp.toSec());

            // April tag data
            cloud_data.push_back(msg);
        }
    }

    // Associate the odom with the closest tag data
    for(int i = 0; i < odom_data.size(); i++)
    {
        double t_odom = odom_data[i]->header.stamp.toSec();
        
        double min_atag_tdiff = -1;
        int closest_atag_idx = -1;
        for(int j = 0; j < atag_data.size(); j++)
        {
            double t_atag = atag_data[j]->header.stamp.toSec();
            double time_diff = t_odom - t_atag;

            if (closest_atag_idx == -1 || fabs(time_diff) < min_atag_tdiff)
            {
                min_atag_tdiff = fabs(time_diff);
                closest_atag_idx = j;
            }

            // If atag time is much later than the odom, abandon the search
            if (time_diff < -1.0)
                break;
        }
        odomMsg tag_data = nullptr;    
        if (closest_atag_idx != -1 && min_atag_tdiff < tag_time_tolerance)
            tag_data = atag_data[closest_atag_idx];

        double min_cloud_tdiff = -1;
        int closest_cloud_idx = -1;
        for(int j = 0; j < cloud_data.size(); j++)
        {
            double t_cloud = cloud_data[j]->header.stamp.toSec();
            double time_diff = t_odom - t_cloud;

            if (closest_cloud_idx == -1 || fabs(time_diff) < min_cloud_tdiff)
            {
                min_cloud_tdiff = fabs(time_diff);
                closest_cloud_idx = j;
            }

            // If atag time is much later than the odom, abandon the search
            if (time_diff < -1.0)
                break;
        }
        cloudMsg pc_data = nullptr;
        if (closest_cloud_idx != -1 && min_cloud_tdiff < 0.01)
            pc_data = cloud_data[closest_cloud_idx];    

        SyncedData synced_data;
        synced_data.odom_data  = odom_data[i];
        synced_data.atag_data  = tag_data;
        synced_data.cloud_data = pc_data;
        synchronized_data.push_back(synced_data);

        if (tag_data != nullptr && pc_data != nullptr)
            printf("odom time %f. Sync: %f, %f\n", t_odom, min_atag_tdiff, min_cloud_tdiff);
    }

    ros::Publisher odom_pub  = nh.advertise<nav_msgs::Odometry>("/odom", 100);
    ros::Publisher atag_pub  = nh.advertise<nav_msgs::Odometry>("/atag_inW", 100);
    ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud", 100);

    static tf::TransformBroadcaster tfbr;

    printf("Pairs: %d. SyncCount: %d\n", synchronized_data.size());
    for(int i = 0; i < synchronized_data.size(); i++)
    {
        odomMsg odom_data = synchronized_data[i].odom_data;
        myTf tf_W_B(*odom_data);

        if (odom_data != nullptr)
        {
            odom_pub.publish(*odom_data);

            tf::Transform transform;
            transform.setOrigin(tf::Vector3(tf_W_B.pos(0), tf_W_B.pos(1), tf_W_B.pos(2)));
            transform.setRotation(tf::Quaternion(tf_W_B.rot.w(), tf_W_B.rot.x(), tf_W_B.rot.y(), tf_W_B.rot.z()));
            tfbr.sendTransform(tf::StampedTransform(transform, odom_data->header.stamp, odom_data->header.frame_id, odom_data->child_frame_id));
        }

        odomMsg atag_data = synchronized_data[i].atag_data;
        if (synchronized_data[i].atag_data != nullptr)
        {
            myTf tf_B_T(*atag_data);
            myTf tf_x_z(Util::YPR2Quat(Vector3d(0, 90, 0)), Vector3d(0, 0, 0));

            printf("Tag pose: %9.3f, %9.3f, %9.3f. Time diff: %6.3f\n",
                    atag_data->pose.pose.position.x,
                    atag_data->pose.pose.position.y,
                    atag_data->pose.pose.position.z,
                   (atag_data->header.stamp - odom_data->header.stamp).toSec());

            // transform the atag data into world frame
            myTf tf_W_T = tf_W_B*tf_B_T*tf_x_z;
            
            nav_msgs::Odometry tagInW = *atag_data;

            tagInW.header.frame_id = "world";
            tagInW.child_frame_id = "tag";

            tagInW.pose.pose.position.x = tf_W_T.pos(0);
            tagInW.pose.pose.position.y = tf_W_T.pos(1);
            tagInW.pose.pose.position.z = tf_W_T.pos(2);

            tagInW.pose.pose.orientation.x = tf_W_T.rot.x();
            tagInW.pose.pose.orientation.y = tf_W_T.rot.y();
            tagInW.pose.pose.orientation.z = tf_W_T.rot.z();
            tagInW.pose.pose.orientation.w = tf_W_T.rot.w();
            atag_pub.publish(tagInW);

            // Also broadcast a tf for easy viewing
        }

        cloudMsg cloud_data = synchronized_data[i].cloud_data;
        if (cloud_data != nullptr)
            cloud_pub.publish(*cloud_data);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // return 0;
    ros::Rate rate(1);
    while(ros::ok())
        rate.sleep();

}