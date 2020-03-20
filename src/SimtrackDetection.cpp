/* Copyright (c) 2012, Ferenc Balint-Benczed<balintbe@cs.uni-bremen.de>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Institute for Aritficial Intelligence/
 *       University of Bremen nor the names of its contributors
 *       may be used to endorse or promote products derived from this software
 *       without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <uima/api.hpp>

#include <ctype.h>

#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include <robosherlock/utils/output.h>
#include <robosherlock/DrawingAnnotator.h>
#include <robosherlock/utils/common.h>

#include <simtrack_nodes/GetDetections.h>
#include <simtrack_nodes/SwitchObjects.h>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <tf_conversions/tf_eigen.h>
using namespace uima;

class SimtrackDetection : public DrawingAnnotator
{
  typedef pcl::PointXYZRGBA PointT;

private:
  cv::Mat color;

  ros::NodeHandle nh;
  tf::StampedTransform camToWorld, worldToCam;
  ros::ServiceClient client_;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudPtr;
  std::vector<std::vector<int>> detectionIndices;
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> bbClouds;
  double pointSize;


  void createImageRoi(const pcl::PointIndices &indices,
                      cv::Rect &roi, cv::Rect &roiHires,
                      cv::Mat &mask, cv::Mat &maskHires)
  {


    size_t width = cloudPtr->width;
    size_t height = cloudPtr->height;

    int min_x = width;
    int max_x = -1;
    int min_y = height;
    int max_y = -1;

    cv::Mat mask_full = cv::Mat::zeros(height, width, CV_8U);

    // get min / max extents (rectangular bounding box in image (pixel) coordinates)
    //#pragma omp parallel for
    for(size_t i = 0; i < indices.indices.size(); ++i)
    {
      const int idx = indices.indices[i];
      const int x = idx % width;
      const int y = idx / width;

      min_x = std::min(min_x, x);
      min_y = std::min(min_y, y);
      max_x = std::max(max_x, x);
      max_y = std::max(max_y, y);

      mask_full.at<uint8_t>(y, x) = 255;
    }

    roi = cv::Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
    roiHires = cv::Rect(roi.x << 1, roi.y << 1, roi.width << 1, roi.height << 1);
    mask_full(roi).copyTo(mask);
    cv::resize(mask, maskHires, cv::Size(0, 0), 2.0, 2.0, cv::INTER_NEAREST);
  }

public:

  SimtrackDetection() : DrawingAnnotator(__func__), nh("~"),
    cloudPtr(new pcl::PointCloud<pcl::PointXYZRGBA>), pointSize(1.0)
  {
    client_ = nh.serviceClient<simtrack_nodes::GetDetections>("/simtrack/get_detected_objects");
  }

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");
    return UIMA_ERR_NONE;
  }

  TyErrorId typeSystemInit(TypeSystem const &type_system)
  {
    outInfo("typeSystemInit");
    return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }

private:
  TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec)
  {
    outInfo("process begins");
    rs::StopWatch clock;
    simtrack_nodes::GetDetections srv;

    if(client_.call(srv))
    {
      if(!srv.response.detections.empty())
      {
        detectionIndices.clear();

        bbClouds.clear();
        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();
        camToWorld.setIdentity();

        cas.get(VIEW_CLOUD, *cloudPtr);
        if(scene.viewPoint.has())
        {
          rs::conversion::from(scene.viewPoint.get(), camToWorld);
        }
        else
        {
          outInfo("No camera to world transformation!!!");
        }
        worldToCam = tf::StampedTransform(camToWorld.inverse(),
                                          camToWorld.stamp_,
                                          camToWorld.child_frame_id_,
                                          camToWorld.frame_id_);

        for(int i = 0; i < srv.response.detections.size(); ++i)
        {
          simtrack_nodes::SimtrackDetection &simtrackDetection = srv.response.detections.at(i);
          outInfo("Found object: " << simtrackDetection.model_name);
          rs::ObjectHypothesis simtrackCluster = rs::create<rs::ObjectHypothesis>(tcas);
          simtrackCluster.source.set("Simtrack");
          rs::PoseAnnotation poseAnnotation = rs::create<rs::PoseAnnotation>(tcas);
          rs::Detection detection = rs::create<rs::Detection>(tcas);

          detection.source.set("Simtrack");
          detection.confidence.set(1.0);
          detection.name.set(simtrackDetection.model_name);

          tf::Stamped<tf::Pose> poseCam;
          tf::poseStampedMsgToTF(simtrackDetection.pose, poseCam);

          tf::Stamped<tf::Pose> poseWorld(poseCam * camToWorld, camToWorld.stamp_, camToWorld.frame_id_);
          poseAnnotation.camera.set(rs::conversion::to(tcas, poseCam));
          poseAnnotation.world.set(rs::conversion::to(tcas, poseWorld));
          poseAnnotation.source.set("Simtrack");

          simtrackCluster.annotations.append(detection);
          simtrackCluster.annotations.append(poseAnnotation);

          Eigen::Affine3d eigenTransform;
          tf::transformTFToEigen(poseCam.inverse(), eigenTransform);

          pcl::PointCloud<PointT>::Ptr transformedCloud(new pcl::PointCloud<PointT>());
          pcl::transformPointCloud<PointT>(*cloudPtr, *transformedCloud, eigenTransform);

          pcl::PointCloud<pcl::PointXYZ>::Ptr bb_cloud(new pcl::PointCloud<pcl::PointXYZ>);
          pcl::PointCloud<pcl::PointXYZ>::Ptr bb_cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
          for(int point_idx = 0; point_idx < simtrackDetection.bb_points.size(); ++point_idx)
          {
            geometry_msgs::Point bb_point = simtrackDetection.bb_points.at(point_idx);
            pcl::PointXYZ point;
            point.x = bb_point.x;
            point.y = bb_point.y;
            point.z = bb_point.z;
            bb_cloud->points.push_back(point);
          }
          bbClouds.push_back(bb_cloud);

          pcl::transformPointCloud<pcl::PointXYZ>(*bb_cloud, *bb_cloud_transformed, eigenTransform);

          Eigen::Vector4f centroid;
          pcl::compute3DCentroid(*bb_cloud_transformed, centroid);
          double scale_factor = 0.25;
          for(size_t i = 0; i < bb_cloud_transformed->points.size(); ++i)
          {
            Eigen::Vector4f scaled_vector = (bb_cloud_transformed->points[i].getVector4fMap() - centroid) * scale_factor;
            bb_cloud_transformed->points[i].getVector4fMap() += scaled_vector;
          }
          Eigen::Vector4f min_pt;
          Eigen::Vector4f max_pt;

          pcl::getMinMax3D(*bb_cloud_transformed, min_pt, max_pt);
          pcl::PointIndices indices;
          pcl::getPointsInBox(*transformedCloud, min_pt, max_pt, indices.indices);
          outInfo("Indices found: " << indices.indices.size());

          rs::ReferenceClusterPoints rcp = rs::create<rs::ReferenceClusterPoints>(tcas);

          rs::PointIndices uimaIndices = rs::conversion::to(tcas, indices);
          rcp.indices.set(uimaIndices);

          cv::Rect roi, roiHires;
          cv::Mat mask, maskHires;

          createImageRoi(indices, roi, roiHires, mask, maskHires);

          rs::ImageROI imageRoi = rs::create<rs::ImageROI>(tcas);
          imageRoi.mask(rs::conversion::to(tcas, mask));
          imageRoi.mask_hires(rs::conversion::to(tcas, maskHires));
          imageRoi.roi(rs::conversion::to(tcas, roi));
          imageRoi.roi_hires(rs::conversion::to(tcas, roiHires));
          simtrackCluster.rois.set(imageRoi);

          simtrackCluster.points.set(rcp);

          scene.identifiables.append(simtrackCluster);

          detectionIndices.push_back(indices.indices);
        }
      }
    }
    else
    {
      outError("Failed to call service /simtrack/get_detected_objects");
    }
    outInfo(clock.getTime() << " ms.");
    return  UIMA_ERR_NONE;
  }

  void drawImageWithLock(cv::Mat disp)
  {

  }
  void fillVisualizerWithLock(pcl::visualization::PCLVisualizer &visualizer, const bool firstRun)
  {
    const std::string &cloudname = this->name;
    for(size_t i = 0; i < detectionIndices.size(); ++i)
    {
      std::vector<int> &indices = detectionIndices[i];
      for(size_t j = 0; j < indices.size(); ++j)
      {
        size_t index = indices[j];
        cloudPtr->points[index].rgba = rs::common::colors[i % rs::common::numberOfColors];
      }
    }

    if(firstRun)
    {
      visualizer.addPointCloud(cloudPtr, cloudname);
      visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize, cloudname);
      for(int i = 0; i < bbClouds.size(); ++i)
      {
        std::stringstream bbCloudName;
        bbCloudName << "bb_cloud_" << i;
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green_color(bbClouds[i], 0, 255, 0);
        visualizer.addPointCloud(bbClouds[i], green_color, bbCloudName.str());
        visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize * 5, bbCloudName.str());
      }
    }
    else
    {
      visualizer.removeAllPointClouds();
      visualizer.addPointCloud(cloudPtr, cloudname);
      visualizer.getPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize, cloudname);
      for(int i = 0; i < bbClouds.size(); ++i)
      {
        std::stringstream bbCloudName;
        bbCloudName << "bb_cloud_" << i;
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green_color(bbClouds[i], 0, 255, 0);
        visualizer.addPointCloud(bbClouds[i], green_color, bbCloudName.str());
        visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize * 5, bbCloudName.str());
      }

    }
  }

};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(SimtrackDetection)
