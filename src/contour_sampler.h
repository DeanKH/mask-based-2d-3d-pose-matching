#pragma once

#include <glm/glm.hpp>
#include <maskgen/camera.h>
#include <maskgen/mask_generator.h>
#include <maskgen/mesh.h>
#include <maskgen/mask_generator.h>
#include <opencv2/core.hpp>
#include <vector>

namespace pose_matching {

std::vector<glm::vec3> SampleContour3D(
    const maskgen::Mesh& mesh,
    const maskgen::MeshPose& initial_pose,
    const maskgen::CameraParams& camera_params,
    maskgen::MaskGenerator& generator,
    int num_points);

}  // namespace pose_matching
