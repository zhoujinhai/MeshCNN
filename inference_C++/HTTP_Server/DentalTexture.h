#pragma once
#include "../../core/mesh/common.h"
#include <string>


namespace core
{
	class Entity;
	class Mesh;
	class LDNI;
	class Image;
	class VoxelObject;
}

namespace dental
{
	class ClassifyByAI
	{
	public:
		ClassifyByAI();
		~ClassifyByAI();
		bool LoadModel(const std::string onnxModelPath);
		bool Predict(core::Image& image, int& classId, int imgSize = 64);
	private:
		void PreProcess(core::Image& image, int imgSize = 64);
		struct CPrivate;
		CPrivate* const mpD;
	};

	class DentalTexture
	{
	public:
		DentalTexture();
		~DentalTexture();
		static bool GetToothTexture(core::Mesh& mesh, const std::string AIModelPath, std::vector<std::vector<int> >& textureFaces);
		static bool GetMinCruvPts(core::Mesh& mesh, std::vector<core::Vector3>& minCurvPts, std::vector<core::Vector3>& ptNormals, double minCruvThreshold = -0.50);

		enum class ReqMode { UPLOAD_FILE, JSON_DATA };
		static bool GetToothTextureHTTP(core::Mesh& mesh, std::vector<core::Vector3>& texturePts, std::vector<core::Vector3>& ptsNormals, const std::string IP, const std::string Port, bool useNormal = false, std::string route = "/recognize", const ReqMode reqMode = ReqMode::JSON_DATA, double minCruvThreshold = -0.50);
	};
}


