#include <uima/api.hpp>

#include <pcl/point_types.h>
#include <robosherlock/types/all_types.h>
//RS
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include <json_prolog/prolog.h>

using namespace uima;


class BeliefToKnowRob : public Annotator
{
private:

public:

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");
    return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }

  std::string buildPerceivecAtQuery(const  tf::Stamped<tf::Pose> &p, const std::string &name)
  {
    std::stringstream ss;
    ss << "belief_perceived_at(kitchen:'" << name << "',";
    ss << "['" << p.frame_id_ << "',_,"
       << "[" << p.getOrigin().x() << "," << p.getOrigin().y() << "," << p.getOrigin().z() << "],"
       << "[" << p.getRotation().x() << "," << p.getRotation().y() << "," << p.getRotation().z() << "," << p.getRotation().w() << "]],"
       << "0.2, ID)";
    return ss.str();
  }

  //  belief_perceived_at(knowrob:'Cup', [ReferenceFrame, _, Translation, Rotation],
  //                      [ThresholdTranslation, ThresholdRotation], ExistingOrNewCupObjectId)
  TyErrorId process(CAS &tcas, ResultSpecification const &res_spec)
  {
    outInfo("process start");
    rs::StopWatch clock;
    rs::SceneCas cas(tcas);
    rs::Scene scene = cas.getScene();
    uint64_t now = scene.timestamp();

    std::vector<rs::Object> objects;
    cas.get(VIEW_OBJECTS, objects);
    if(objects.empty())
    {
      outWarn("There are no objects in the perceptual BeliefState;");
      return UIMA_ERR_NONE;
    }
    int idx = 0;

    json_prolog::Prolog pl;

    for(rs::Object &obj : objects)
    {
      double lastSeen = (now - (uint64_t)obj.lastSeen()) / 1000000000.0;
      if(lastSeen != 0)
        continue;
      std::vector<rs::Classification> classificationResult;
      obj.annotations.filter(classificationResult);
      if(classificationResult.empty())
      {
        continue;
      }
      std::string name = classificationResult[0].classname();

      std::vector<rs::Geometry> geom;
      obj.annotations.filter(geom);
      std::vector<rs::PoseAnnotation> poses;
      obj.annotations.filter(poses);
      if(geom.empty() || poses.empty())
      {
        continue;
      }
      rs::Geometry &g = geom[0];
      rs::PoseAnnotation &p = poses[0];
      tf::Stamped<tf::Pose> pose;
      rs::conversion::from(p.world(), pose);
      std::string plQuery = buildPerceivecAtQuery(pose, name);
      outInfo(plQuery);
      json_prolog::PrologQueryProxy bdgs = pl.query(plQuery);
      for(auto bdg : bdgs)
      {
        outInfo(bdg["ID"].toString());
        std::string uid = bdg["ID"].toString();
        obj.uid.set(uid.substr(1, uid.size() - 2));
        break;
      }
      idx ++;
    }
    outInfo("Process ends");
    return UIMA_ERR_NONE;
  }
};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(BeliefToKnowRob)
