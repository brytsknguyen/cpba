#include <thread>     // Include the <thread> header for sleep_for
#include <chrono>     // Include the <chrono> header for time durations

#include "ros/ros.h"
#include <rosbag/view.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

#include "nav_msgs/Odometry.h"
#include "sensor_msgs/PointCloud2.h"

#include "utility.h"

// Ceres
#include <ceres/ceres.h>
#include "factor/PoseFactorAnalytic.h"
#include "PoseLocalParameterization.h"

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
    odomMsg atag_filtered_data;
    cloudMsg cloud_data;
};

struct atagCluster
{
    atagCluster(odomMsg atag_data_, int odom_idx_)
    {
        atag_data.push_back(atag_data_);
        odom_idx.push_back(odom_idx_);
    }
    
    void push_back(odomMsg atag_data_, int odom_idx_)
    {
        atag_data.push_back(atag_data_);
        odom_idx.push_back(odom_idx_);
    }

    int size()
    {
        return atag_data.size();
    }

    deque<odomMsg> atag_data;
    deque<int> odom_idx;
};

struct RelocStat
{
    int keyframe_id = -1;
    double t = -1;
    myTf<double> tf = myTf<double>();
    Vector3d p_var = Vector3d(0, 0, 0);
    Vector3d r_var = Vector3d(0, 0, 0);
    double PDM = -1; // positional mahalanobis distance
    double RDM = -1; // rotational mahalanobis distance
    double ceres_cost = -1;
    bool inlier = false;
};

deque<odomMsg> atag_buffer;
deque<odomMsg> odom_buffer;
deque<cloudMsg> cloud_buffer;
deque<SyncedData> synchronized_data;

double odom_tag_time_tolerance;
double cluster_timeout = 5.0;
double outlier_thres = 0.5;

void clusterAtagData(deque<SyncedData> &ynchronized_data, deque<atagCluster> &cluster)
{
    double cluster_end_time = -1;
    for(int i = 0; i < synchronized_data.size(); i++)
    {
        if(synchronized_data[i].atag_data == nullptr)
            continue;

        double atag_time = synchronized_data[i].atag_data->header.stamp.toSec();

        // If cluster start_time is -1, start a new cluster
        if (cluster_end_time == -1)
        {
            atagCluster newCluster(synchronized_data[i].atag_data, i);
            cluster_end_time = atag_time;
            cluster.push_back(newCluster);
            continue;
        }

        if (fabs(atag_time - cluster_end_time) < cluster_timeout)
        {
            cluster_end_time = atag_time;
            cluster.back().push_back(synchronized_data[i].atag_data, i);
            continue;
        }
        else
        {
            atagCluster newCluster(synchronized_data[i].atag_data, i);
            cluster_end_time = atag_time;
            cluster.push_back(newCluster);
            continue;
        }
    }
}

mytf OptimizePoseWithCeres(Vector3d p_init, Quaternd q_init, deque<RelocStat> &relocStat)
{
    // Create and solve the Ceres Problem
    ceres::Problem problem;
    ceres::Solver::Options options;

    // Set up the options
    // options.minimizer_type = ceres::TRUST_REGION;
    options.linear_solver_type           = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type   = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations           = 40;
    options.max_solver_time_in_seconds   = 1.0;
    options.num_threads                  = MAX_THREADS;
    options.minimizer_progress_to_stdout = false;

    // Create optimization params
    double* PARAM_POSE = new double[7];
    
    PARAM_POSE[0] = p_init(0);
    PARAM_POSE[1] = p_init(1);
    PARAM_POSE[2] = p_init(2);

    PARAM_POSE[3] = q_init.x();
    PARAM_POSE[4] = q_init.y();
    PARAM_POSE[5] = q_init.z();
    PARAM_POSE[6] = q_init.w();

    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(PARAM_POSE, 7, local_parameterization);

    vector<ceres::internal::ResidualBlock *> res_ids_pose;
    double cost_pose_init = -1, cost_pose_final = -1;
    for(RelocStat &stat : relocStat)
    {
        ceres::LossFunction *loss_function = new ceres::ArctanLoss(1.0);
        PoseFactorAnalytic *f = new PoseFactorAnalytic(stat.tf.pos, stat.tf.rot, 100.0, 100.0);
        ceres::internal::ResidualBlock *res_id = problem.AddResidualBlock(f, loss_function, PARAM_POSE);
        res_ids_pose.push_back(res_id);
    }

    Util::ComputeCeresCost(res_ids_pose, cost_pose_init, problem);

    TicToc tt_solve;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    tt_solve.Toc();

    Util::ComputeCeresCost(res_ids_pose, cost_pose_final, problem);

    Vector3d pos_est(PARAM_POSE[0], PARAM_POSE[1], PARAM_POSE[2]);
    Quaternd rot_est(PARAM_POSE[6], PARAM_POSE[3], PARAM_POSE[4], PARAM_POSE[5]);
    myTf tf_est(rot_est, pos_est);
    myTf tf_ini(q_init, p_init);

    printf("Tslv: %.3f. Iter: %d. J0: %6.3f -> JK: %6.3f. \n"
            "PoseEstStart: %9.9f, %9.9f, %9.9f, %9.9f, %9.9f, %9.9f.\n"
            "PoseEstFinal: %9.9f, %9.9f, %9.9f, %9.9f, %9.9f, %9.9f.\n",
            tt_solve.Toc(), summary.iterations.size(),
            cost_pose_init, cost_pose_final,
            tf_ini.pos(0), tf_ini.pos(1), tf_ini.pos(2), tf_ini.yaw(), tf_ini.pitch(), tf_ini.roll(),
            tf_est.pos(0), tf_est.pos(1), tf_est.pos(2), tf_est.yaw(), tf_est.pitch(), tf_est.roll());

    for(int i = 0; i < relocStat.size(); i++)
    {
        vector<ceres::internal::ResidualBlock *> res(1, res_ids_pose[i]);
        double cost;
        Util::ComputeCeresCost(res, cost, problem);
        relocStat[i].ceres_cost = cost;
    }

    return tf_est;
}

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

    nh_ptr->param("odom_tag_time_tolerance", odom_tag_time_tolerance, 0.05);
    nh_ptr->param("cluster_timeout", cluster_timeout, 2.0);
    nh_ptr->param("outlier_thres", outlier_thres, 0.5);

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
            atag_buffer.push_back(msg);
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
            odom_buffer.push_back(msg);
        }

        if (m.getTopic() == "/cloud_registered")
        {           
            // Load the data into the tag buffer
            sensor_msgs::PointCloud2::Ptr msg = m.instantiate<sensor_msgs::PointCloud2>();
            msg->header.frame_id = "world";

            // Process the message data
            // ROS_INFO("Topic: %s. Time: %f", m.getTopic().c_str(), msg->header.stamp.toSec());

            // April tag data
            cloud_buffer.push_back(msg);
        }
    }

    synchronized_data.resize(odom_buffer.size());

    // Associate the odom with the closest tag data
    #pragma omp parallel for num_threads(MAX_THREADS)
    for(int i = 0; i < odom_buffer.size(); i++)
    {
        double t_odom = odom_buffer[i]->header.stamp.toSec();
        
        double min_atag_tdiff = -1;
        int closest_atag_idx = -1;
        for(int j = 0; j < atag_buffer.size(); j++)
        {
            double t_atag = atag_buffer[j]->header.stamp.toSec();
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
        if (closest_atag_idx != -1 && min_atag_tdiff < odom_tag_time_tolerance)
            tag_data = atag_buffer[closest_atag_idx];

        double min_cloud_tdiff = -1;
        int closest_cloud_idx = -1;
        for(int j = 0; j < cloud_buffer.size(); j++)
        {
            double t_cloud = cloud_buffer[j]->header.stamp.toSec();
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
            pc_data = cloud_buffer[closest_cloud_idx];    

        SyncedData synced_data;
        synced_data.odom_data  = odom_buffer[i];
        synced_data.atag_data  = tag_data;
        synced_data.atag_filtered_data = nullptr;
        synced_data.cloud_data = pc_data;
        synchronized_data[i] = synced_data;

        // if (tag_data != nullptr && pc_data != nullptr)
        //     printf("odom time %f. Sync: %f, %f\n", t_odom, min_atag_tdiff, min_cloud_tdiff);
    }

    deque<atagCluster> cluster;
    clusterAtagData(synchronized_data, cluster);
    for(int i = 0; i < cluster.size(); i++)
    {
        // printf("Cluster %d. Size: %d. Ids:\n\t", i, cluster[i].atag_data.size());
        // for(auto &data : cluster[i].atag_data)
        //     cout << data->pose.covariance[0] << "," ;
        // printf("\n");
        
        // If cluster is too small, discard these
        if (cluster[i].size() < 5)
        {
            printf("Clearing cluster %d\n", i);
            for(auto &idx : cluster[i].odom_idx)
                synchronized_data[idx].atag_data = nullptr;
        }
    }

    // Redo the clustering
    cluster.clear();
    clusterAtagData(synchronized_data, cluster);

    // Filter the tag poses
    for(int i = 0; i < cluster.size(); i++)
    {
        printf("Filtering cluster %d / %d. Size: %d, %d. Ids: ", i, cluster.size(), cluster[i].atag_data.size(), cluster[i].size());
        for(auto &data : cluster[i].atag_data)
            cout << data->pose.covariance[0] << "," ;
        printf("\n");

        deque<RelocStat> relocStat;

        // Convert the pose points to pose tfs
        for(int j = 0; j < cluster[i].size(); j++)
        {

            odomMsg odom_data = synchronized_data[cluster[i].odom_idx[j]].odom_data;
            myTf tf_W_B(*odom_data);
            
            odomMsg atag_odom = cluster[i].atag_data[j];
            myTf tf_W_T = tf_W_B*myTf(*atag_odom);
            
            RelocStat stat;
            stat.keyframe_id = atag_odom->pose.covariance[0];
            stat.t  = atag_odom->header.stamp.toSec();
            stat.tf = tf_W_T;
            relocStat.push_back(stat);
        }

        int N = cluster[i].atag_data.size();

        // Filter the cluster
        Vector3d p_mean(0, 0, 0);
        Vector3d r_mean(0, 0, 0);
        for(RelocStat &stat : relocStat)
        {
            p_mean += stat.tf.pos;
            r_mean += stat.tf.SO3Log();
        }
        p_mean /= N;
        r_mean /= N;
        Quaternd q_mean(Eigen::AngleAxis<double>(r_mean.norm(), r_mean/r_mean.norm()));

        // printf("Mean Pos: %f, %f, %f. Phi: %f, %f, %f\n", p_mean(0), p_mean(1), p_mean(2), r_mean(0), r_mean(1), r_mean(2));

        // Optimize the pose
        OptimizePoseWithCeres(p_mean, q_mean, relocStat);

        // Sort the reloc by cost
        struct compareRes
        {
            bool const operator()(RelocStat a, RelocStat b) const
            {
                return (a.ceres_cost < b.ceres_cost);
            }
        };
        std::sort(relocStat.begin(), relocStat.end(), compareRes());

        double cost_thres = relocStat[(int)(N*outlier_thres)].ceres_cost;

        // Find the inliear relocalized poses
        deque<RelocStat> relocStatInlier;
        Vector3d p_inlr(0, 0, 0);
        Vector3d r_inlr(0, 0, 0);
        int Ninlr = 0; int count = -1;
        for(RelocStat &stat : relocStat)
        {
            count++;
            printf("cost %3d: %f / %f\n", count, stat.ceres_cost, cost_thres);
            if (stat.ceres_cost < cost_thres)
            {
                Eigen::AngleAxis<double>e(stat.tf.rot);

                p_inlr += stat.tf.pos;
                r_inlr += e.angle()*e.axis();
                Ninlr  += 1;

                relocStatInlier.push_back(stat);
            }
        }
        p_inlr /= Ninlr;
        r_inlr /= Ninlr;
        Quaternd q_inlr(Eigen::AngleAxis<double>(r_inlr.norm(), r_inlr/r_inlr.norm()));

        printf("Inliers: %d\n", Ninlr);

        mytf tf_W_T_opt = OptimizePoseWithCeres(p_inlr, q_inlr, relocStatInlier);

        // Copy the optimized pose back to buffer
        for(int j = 0; j < cluster[i].size(); j++)
        {
            int odom_idx = cluster[i].odom_idx[j];

            odomMsg odom_data = synchronized_data[odom_idx].odom_data;
            myTf tf_W_B(*odom_data);
            myTf tf_W_T = tf_W_B.inverse()*tf_W_T_opt;

            synchronized_data[odom_idx].atag_filtered_data = odomMsg(new nav_msgs::Odometry());
           *synchronized_data[odom_idx].atag_filtered_data = *synchronized_data[odom_idx].atag_data;

            synchronized_data[odom_idx].atag_filtered_data->pose.pose.position.x = tf_W_T.pos(0);
            synchronized_data[odom_idx].atag_filtered_data->pose.pose.position.y = tf_W_T.pos(1);
            synchronized_data[odom_idx].atag_filtered_data->pose.pose.position.z = tf_W_T.pos(2);

            synchronized_data[odom_idx].atag_filtered_data->pose.pose.orientation.x = tf_W_T.rot.x();
            synchronized_data[odom_idx].atag_filtered_data->pose.pose.orientation.y = tf_W_T.rot.y();
            synchronized_data[odom_idx].atag_filtered_data->pose.pose.orientation.z = tf_W_T.rot.z();
            synchronized_data[odom_idx].atag_filtered_data->pose.pose.orientation.w = tf_W_T.rot.w();
        }

        printf("\n");
    }

    // Clear the buffers to free the space
    odom_buffer.clear();
    atag_buffer.clear();

    ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("/odom", 100);
    ros::Publisher atag_pub = nh.advertise<nav_msgs::Odometry>("/atag_inW", 100);
    ros::Publisher atag_filtered_pub = nh.advertise<nav_msgs::Odometry>("/atag_filtered_inW", 100);
    ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud", 100);
    static tf::TransformBroadcaster tfbr;

    // Publish the synchronized data for visualization
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
        if (atag_data != nullptr)
        {
            {
                myTf tf_B_T(*atag_data);
                myTf tf_x_z(Util::YPR2Quat(Vector3d(0, 90, 0)), Vector3d(0, 0, 0));

                // printf("Tag pose: %9.3f, %9.3f, %9.3f. Time diff: %6.3f\n",
                //         atag_data->pose.pose.position.x,
                //         atag_data->pose.pose.position.y,
                //         atag_data->pose.pose.position.z,
                //        (atag_data->header.stamp - odom_data->header.stamp).toSec());

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
            }
        }

        odomMsg atag_filtered_data = synchronized_data[i].atag_filtered_data;
        if (atag_filtered_data != nullptr)
        {
            {
                myTf tf_B_T(*atag_filtered_data);
                myTf tf_x_z(Util::YPR2Quat(Vector3d(0, 90, 0)), Vector3d(0, 0, 0));

                // printf("Tag pose: %9.3f, %9.3f, %9.3f. Time diff: %6.3f\n",
                //         atag_data->pose.pose.position.x,
                //         atag_data->pose.pose.position.y,
                //         atag_data->pose.pose.position.z,
                //        (atag_data->header.stamp - odom_data->header.stamp).toSec());

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
                atag_filtered_pub.publish(tagInW);
            }
        }

        cloudMsg cloud_data = synchronized_data[i].cloud_data;
        if (cloud_data != nullptr)
            cloud_pub.publish(*cloud_data);

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // return 0;
    ros::Rate rate(1);
    while(ros::ok())
        rate.sleep();

}