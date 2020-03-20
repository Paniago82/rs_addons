#include <uima/api.hpp>

#include <pcl/point_types.h>

//RS
#include <robosherlock/scene_cas.h>
#include <robosherlock/types/all_types.h>
#include <robosherlock/DrawingAnnotator.h>
#include <robosherlock/utils/time.h>
#include <robosherlock/utils/output.h>

//Caffe
#include <caffe/caffe.hpp>

#include <ros/package.h>

#include <robosherlock/recognition/CaffeProxy.h>

typedef pcl::PointXYZRGBA PointT;
using namespace std;
using namespace uima;
using namespace cv;


class CaffeClassificationAnnotator : public DrawingAnnotator
{
private:

  cv::Mat color;
  std::vector<rs::ObjectHypothesis> clusters;
  std::vector<cv::Rect> clusterRois;
  cv::Mat caffeImage;
  std::vector<std::vector<Prediction>> predictionsAllClusters;

  /**
   * Four arguments needed by the CaffeClasifier object from this annotator
   */
  string resourcesPath;
  string model_file;//caffe prototext model def.
  string trained_file;//the trained model
  string mean_file;//mean images
  string label_file;//sysnsets

  std::shared_ptr<CaffeProxy> classifier;

  float test_param;
  double pointSize;

  pcl::PointCloud<PointT>::Ptr cloud_ptr;

public:

  CaffeClassificationAnnotator() : DrawingAnnotator(__func__), pointSize(1),
    cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>)
  {}

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");
    resourcesPath = ros::package::getPath("rs_resources") + '/';

    ctx.extractValue("model_file", model_file);
    ctx.extractValue("trained_file", trained_file);
    ctx.extractValue("mean_file", mean_file);
    ctx.extractValue("label_file", label_file);
    if(!boost::filesystem::exists(resourcesPath + model_file) ||
       !boost::filesystem::exists(resourcesPath + trained_file) ||
       !boost::filesystem::exists(resourcesPath + mean_file) ||
       !boost::filesystem::exists(resourcesPath + label_file))
    {
      outError("files not found!");
      return UIMA_ERR_USER_ANNOTATOR_COULD_NOT_INIT;
    }
    classifier = std::make_shared<CaffeProxy>(resourcesPath + model_file,
                                resourcesPath + trained_file,
                                resourcesPath + mean_file,
                                resourcesPath + label_file);
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

  TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec)
  {
    predictionsAllClusters.clear();
    outInfo("process start");
    rs::StopWatch clock;
    rs::SceneCas cas(tcas);
    rs::Scene scene = cas.getScene();

    cas.get(VIEW_CLOUD, *cloud_ptr);
    clusters.clear();
    cas.get(VIEW_COLOR_IMAGE_HD, color);

    //2.filter out clusters into array
    scene.identifiables.filter(clusters);
    clusterRois.resize(clusters.size());
    outInfo("Number of clusters:" << clusters.size());

    cv::Mat tempImage(900, 1680, CV_8UC3, Scalar(170, 170, 170));

    //current width and height
    int crtX = 20;
    int crtY = 30;
    // Extract ROIS from RGB image
    for(size_t idx = 0; idx < clusters.size(); idx++)
    {
      rs::ImageROI image_rois = clusters[idx].rois.get();
      cv::Rect roi;
      rs::conversion::from(image_rois.roi_hires(), roi);
      clusterRois[idx] = roi;

      int padding = 20;
      roi.x -= padding;
      roi.y -= padding;
      roi.width += 2 * padding;
      roi.height += 2 * padding;

      cv::Size s = color(roi).size();
      cv::Rect roiCpy(cv::Point(crtX, crtY), color(roi).size());
      cv::Mat inputImage = color(roi).clone();


      std::vector<Prediction> predictions = classifier->Classify(inputImage);

      std::vector<float> feature = classifier->extractFeature(inputImage);
      outInfo("Size of feature extracted is: " << feature.size());
      outInfo("max element is : " << *std::max_element(feature.begin(), feature.end()));

      predictionsAllClusters.push_back(predictions);
      ////////////////////////////////////////////////////////////////////

      color(roi).copyTo(tempImage(roiCpy));

      int tempY = crtY;
      for(int ii = 0; ii < predictions.size(); ++ii)
      {
        Prediction p = predictions[ii];
        std::string tempStr = p.first;
        tempStr = tempStr.substr(tempStr.find(" "), tempStr.size() - 1);
        float tempFloat = ceilf(100 * p.second) / 100;
        std::string tempStr1 = std::to_string(tempFloat);
        tempStr1 = tempStr1.substr(0, 4) ;

        if(tempFloat >= 0.3)
        {
          putText(tempImage, (tempStr1 + tempStr), cvPoint(5 + s.width + crtX, tempY + 15),
                  FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);

        }
        else
        {
          putText(tempImage, (tempStr1 + tempStr), cvPoint(5 + s.width + crtX, tempY + 15),
                  FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(102, 0, 102), 1, CV_AA);
        }
        tempY += 15;
      }


      if(crtY + 2 * s.height < tempImage.rows)
      {
        crtY += s.height + 15;
      }
      else
      {
        crtY = 30;
        crtX += 560;
      }

    }


    caffeImage = tempImage.clone();
    outInfo("Cloud size: " << cloud_ptr->points.size());
    outInfo("took: " << clock.getTime() << " ms.");
    return UIMA_ERR_NONE;

  }



  void drawImageWithLock(cv::Mat &disp)
  {
    disp = caffeImage.clone();
  }

};

MAKE_AE(CaffeClassificationAnnotator)


























