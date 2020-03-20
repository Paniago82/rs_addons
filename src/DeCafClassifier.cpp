#include <uima/api.hpp>

#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/shot_omp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/fpfh.h>

#include <flann/flann.h>
#include <flann/io/hdf5.h>

//Caffe
#include <caffe/caffe.hpp>

//RS
#include <robosherlock/types/all_types.h>
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include <robosherlock/DrawingAnnotator.h>

#include <ros/package.h>

#include <robosherlock/recognition/CaffeProxy.h>


using namespace uima;

class DeCafClassifier : public DrawingAnnotator
{
private:
  typedef std::pair<std::string, std::vector<float> > Model;
  typedef flann::Index<flann::ChiSquareDistance<float> > Index;

  std::string resourcesPath;
  std::string h5_file, list_file, kdtree_file;
  std::string caffe_model_file, caffe_trained_file, caffe_mean_file, caffe_label_file;
  std::vector<Model> models;

  cv::Mat data;
  std::shared_ptr<CaffeProxy> caffeProxyObj;
  cv::flann::Index index;

  bool asGT;
  int k;

  cv::Mat color;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
public:

  DeCafClassifier(): DrawingAnnotator(__func__), asGT(false)
  {
    cloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
  }

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");
    resourcesPath = ros::package::getPath("rs_resources") + '/';

    ctx.extractValue("DeCafH5File", h5_file);
    ctx.extractValue("DeCafListFile", list_file);
    ctx.extractValue("DeCafKDTreeIndices", kdtree_file);
    ctx.extractValue("DeCafKNeighbors", k);

    ctx.extractValue("caffe_model_file", caffe_model_file);
    ctx.extractValue("caffe_trained_file", caffe_trained_file);
    ctx.extractValue("caffe_mean_file", caffe_mean_file);
    ctx.extractValue("caffe_label_file", caffe_label_file);

    ctx.extractValue("asGT", asGT);

    outInfo(h5_file);
    outInfo(list_file);
    outInfo(kdtree_file);
    outInfo(caffe_model_file);
    outInfo(caffe_trained_file);
    outInfo(caffe_mean_file);
    outInfo(caffe_label_file);

    // Check if the data has already been saved to disk
    if(!boost::filesystem::exists(resourcesPath + h5_file) ||
       !boost::filesystem::exists(resourcesPath + list_file) ||
       !boost::filesystem::exists(resourcesPath + kdtree_file) ||
       !boost::filesystem::exists(resourcesPath + caffe_model_file) ||
       !boost::filesystem::exists(resourcesPath + caffe_trained_file) ||
       !boost::filesystem::exists(resourcesPath + caffe_mean_file) ||
       !boost::filesystem::exists(resourcesPath + caffe_label_file))
    {
      outError("files not found!");
      return UIMA_ERR_USER_ANNOTATOR_COULD_NOT_INIT;
    }
    caffeProxyObj = std::make_shared<CaffeProxy>(resourcesPath + caffe_model_file,
                    resourcesPath + caffe_trained_file,
                    resourcesPath + caffe_mean_file,
                    resourcesPath + caffe_label_file);

    flann::Matrix<float> data;
    loadFileList(models, resourcesPath + list_file);
    flann::load_from_file(data, resourcesPath + h5_file, "training_data");
    outInfo("Training data found. Loaded " << data.rows << " models from " << h5_file << "/" << list_file);

    this->data = cv::Mat(data.rows, data.cols, CV_32F, data.ptr()).clone();
    index.build(this->data, cv::flann::KDTreeIndexParams());

    return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }

  TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec)
  {
    MEASURE_TIME;
    outInfo("process start");
    rs::SceneCas cas(tcas);

    cas.get(VIEW_CLOUD, *cloud);
    cas.get(VIEW_COLOR_IMAGE_HD, color);

    rs::Scene scene = cas.getScene();

    std::vector<rs::ObjectHypothesis> clusters;
    scene.identifiables.filter(clusters);
    for(int i = 0; i < clusters.size(); ++i)
    {
      rs::ObjectHypothesis &cluster = clusters[i];
      if(!cluster.points.has())
      {
        continue;
      }
      const std::string &name = "cluster" + std::to_string(i);
      cv::Rect roi;
      rs::conversion::from(cluster.rois().roi_hires(), roi);

      const cv::Mat &clusterImg = color(roi);


      std::vector<float> featureVec = caffeProxyObj->extractFeature(clusterImg);
      cv::Mat desc(1, featureVec.size(), CV_32F, &featureVec[0]);
      cv::normalize(desc, desc, 1, 0, cv::NORM_L2);
      featureVec.assign((float *)desc.datastart, (float *)desc.dataend);

      Model feature(name, featureVec);

      std::vector<int> k_indices;
      std::vector<float> k_distances;

      nearestKSearch(index, feature, k, k_indices, k_distances);

      outInfo("The closest " << k << " neighbors for cluser " << i << " are:");
      for(int j = 0; j < k; ++j)
      {
        outInfo("    " << j << " - " << models[k_indices[j]].first << " (" << k_indices[j] << ") with a distance of: " << k_distances[j] << " confidence: " << (2 - k_distances[j]) / 2);
      }
      float confidence = (2 - k_distances[0]) / 2;
      if(confidence > 0.5)
      {
        if(!asGT)
        {
          //          rs::Detection detection = rs::create<rs::Detection>(tcas);
          //          detection.name.set(models[k_indices[0]].first);
          //          detection.source.set("DeCafClassifier");
          //          detection.confidence.set(confidence);
          //          cluster.annotations.append(detection);

          rs::Classification c = rs::create<rs::Classification>(tcas);
          c.classname.set(models[k_indices[0]].first);
          c.classifier.set("DeCafClassifier");

          cluster.annotations.append(c);

        }
        else
        {
          rs::GroundTruth gt = rs::create<rs::GroundTruth>(tcas);
          rs::Classification c = rs::create<rs::Classification>(tcas);
          c.classname.set(models[k_indices[0]].first);
          c.classifier.set("GroundTruth");
          gt.classificationGT.set(c);
          cluster.annotations.append(gt);
        }

        rs::ImageROI image_roi = cluster.rois();
        cv::Rect rect;
        rs::conversion::from(image_roi.roi_hires(), rect);
        drawCluster(rect, models[k_indices[0]].first);
      }
    }

    return UIMA_ERR_NONE;
  }
  void drawCluster(cv::Rect roi, const std::string &label)
  {
    cv::rectangle(color, roi, CV_RGB(255, 0, 0));
    int offset = 7;
    int baseLine;
    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_PLAIN, 0.8, 1, &baseLine);
    cv::putText(color, label, cv::Point(roi.x + (roi.width - textSize.width) / 2, roi.y - offset - textSize.height), cv::FONT_HERSHEY_PLAIN, 0.8, CV_RGB(255, 255, 200), 1.0);
  }

  void drawImageWithLock(cv::Mat &disp)
  {
    disp = color.clone();
  }

  void fillVisualizerWithLock(pcl::visualization::PCLVisualizer &visualizer, bool firstRun)
  {
    if(firstRun)
    {
      visualizer.addPointCloud(cloud, "cloud");
      visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "cloud");
    }
    else
    {
      visualizer.updatePointCloud(cloud, "cloud");
      visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "cloud");
    }
  }

  inline void nearestKSearch(cv::flann::Index &index, const Model &model, int k, std::vector<int> &indices, std::vector<float> &distances)
  {
    indices.resize(k);
    distances.resize(k);
    index.knnSearch(model.second, indices, distances, k, cv::flann::SearchParams(512));
  }

  bool loadFileList(std::vector<Model> &models, const std::string &filename)
  {
    ifstream fs;
    fs.open(filename.c_str());
    if(!fs.is_open() || fs.fail())
    {
      return (false);
    }

    std::string line;
    while(!fs.eof())
    {
      getline(fs, line);
      if(line.empty())
      {
        continue;
      }
      Model m;
      m.first = line;
      models.push_back(m);
    }
    fs.close();
    return (true);
  }
};

MAKE_AE(DeCafClassifier)
