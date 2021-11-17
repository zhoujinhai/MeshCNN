#include "pch.h"
#include <queue>
#include <map>
#include <iterator>
#include <fstream>
#include <algorithm>
#include <comdef.h>
#include "DentalTexture.h"
#include "../../core/mesh/mesh.h"
#include "../../core/mesh/MeshHealing.h"
#include "../../core/mesh/MeshSelection.h"
#include "../../core/mesh/meshFunction.h"
#include "../../core/mesh/Logger.h"
#include "../../core/mesh/KDTreeSAH.h"
#include "../../core/ldni/Rasterizer.h"
#include "../../core/ldni/Image.h"
#include "../../core/AI/MeshCNNHTTP.h""
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/dnn.hpp"
#include "json/json.h"

//#ifdef _DEBUG
#include "../../core/mesh/Polyline.h"
#include "../../core/file/STLFileHandler.h"
#include "../../core/file/PTSHandler.h"
#include "../../core/file/PNGHandler.h"
//#endif // _DEBUG


using namespace core;

namespace dental
{
    void CoreImg2CvMat(const core::Image& coreImg, cv::Mat& cvImg) {
        int w = coreImg.GetWidth();
        int h = coreImg.GetHeight();
        if (cvImg.rows != h || cvImg.cols != w || cvImg.type() != CV_8UC1) {
            cvImg = cv::Mat::zeros(h, w, CV_8UC1);
        }
        memcpy(cvImg.data, coreImg.GetData(), w * h);
    }

    void CvMat2CoreImg(const cv::Mat& cvImg, core::Image& coreImg) {
        int w = cvImg.cols;
        int h = cvImg.rows;
        if (coreImg.GetHeight() != h || coreImg.GetWidth() != w || coreImg.GetComponent() != 1) {
            coreImg.Init(w, h);
        }
        memcpy(coreImg.GetData(), cvImg.data, w * h);
    }

    struct ClassifyByAI::CPrivate
    {
        cv::dnn::Net net;
        cv::Mat img;
    };

    ClassifyByAI::ClassifyByAI() : mpD(new CPrivate) 
    {

    }

    ClassifyByAI::~ClassifyByAI() 
    {
        if (mpD) delete mpD;
    }

    bool ClassifyByAI::LoadModel(const std::string onnxModelPath) 
    {
        mpD->net = cv::dnn::readNetFromONNX(onnxModelPath);
        if (mpD->net.empty()) {
            HGAPI_LOG(ClassifyByAI::LoadModel, L"Load Model failed");
            return false;
        }
        return true;
    }

    bool ClassifyByAI::Predict(core::Image & image, int & classId, int imgSize)
    {
        PreProcess(image, imgSize);

        if (!mpD) {
            HGAPI_LOG(ClassifyByAI::Predict, L"Predict failed");
            return false;
        }
        cv::Mat inputBolb = cv::dnn::blobFromImage(mpD->img);  // 也可进行缩放 通道转换 均值等操作
        mpD->net.setInput(inputBolb);
        cv::Mat result = mpD->net.forward();

        double minValue, maxValue;    // 最大值，最小值
        cv::Point  minIdx, maxIdx;    // 最小值坐标，最大值坐标     
        cv::minMaxLoc(result, &minValue, &maxValue, &minIdx, &maxIdx);
        /*std::cout << "maxValue: " << maxValue << "maxIdx: " << maxIdx.x << std::endl;
        std::cout << "minValue: " << minValue << "minIdx: " << minIdx.x << std::endl;*/
        classId = maxIdx.x;
        return true;
    }

    void ClassifyByAI::PreProcess(core::Image& image, int imgSize)
    {
        // find roi
        cv::Mat cvImg, dilateImg;
        CoreImg2CvMat(image, cvImg);
        if (image.GetWidth() == image.GetHeight() && image.GetHeight() == imgSize) {
            mpD->img = cvImg;
        }
        else {
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
            cv::dilate(cvImg, dilateImg, kernel);
            std::vector< std::vector< cv::Point> > contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(dilateImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (contours.size() < 1) {
                mpD->img = cvImg;
            }
            else {
                std::vector< cv::Point>cnt = contours[0];
                for (int i = 1; i < contours.size(); ++i) {
                    if (contours[i].size() > cnt.size()) {
                        cnt = contours[i];
                    }
                }

                cv::Rect rect = cv::boundingRect(cnt);
                mpD->img = cvImg(rect);
            }
        }
       /* cv::imshow("1", mpD->img);
        cv::waitKey(0);*/
        
        // resize and normalize
        mpD->img.convertTo(mpD->img, CV_32FC3);
        cv::cvtColor(mpD->img, mpD->img, cv::COLOR_BGR2RGB);

        cv::resize(mpD->img, mpD->img, cv::Size(imgSize, imgSize));
        mpD->img = mpD->img / 255.0;
        std::vector<float> mean_value{ 0.485, 0.456, 0.406 };
        std::vector<float> std_value{ 0.229, 0.224, 0.225 };
        
        std::vector<cv::Mat> rgbChannels(3);
        cv::split(mpD->img, rgbChannels);
        for (auto i = 0; i < rgbChannels.size(); i++)
        {
            rgbChannels[i] = (rgbChannels[i] - mean_value[i]) / std_value[i];
        }

        cv::merge(rgbChannels, mpD->img);
    }


    bool IsGumMaxLen(core::Mesh& mesh, std::vector<int>& groupFace, double lenThreshold = 15.0) {
        auto& meshIndex = mesh.GetIndex();
        std::vector<core::Vector3> coord = mesh.GetPoint();

        // judge length
        double XMin = std::numeric_limits<double>::max(), XMax = std::numeric_limits<double>::lowest();
        double YMin = std::numeric_limits<double>::max(), YMax = std::numeric_limits<double>::lowest();
        double ZMin = std::numeric_limits<double>::max(), ZMax = std::numeric_limits<double>::lowest();

        std::vector<core::Vector3> points;
        bool isHoleBoundary = false;

        int nHole = 0;
        for (int fId : groupFace) {
            core::Vector3 faceOnePt = coord[meshIndex[3 * fId]];
            double ptX = faceOnePt.x(), ptY = faceOnePt.y(), ptZ = faceOnePt.z();
            XMax = ptX > XMax ? ptX : XMax;
            XMin = ptX < XMin ? ptX : XMin;
            YMax = ptY > YMax ? ptY : YMax;
            YMin = ptY < YMin ? ptY : YMin;
            ZMax = ptZ > ZMax ? ptZ : ZMax;
            ZMin = ptZ < ZMin ? ptZ : ZMin;
        }
        double XDiff = XMax - XMin;
        double YDiff = YMax - YMin;
        double ZDiff = ZMax - ZMin;

        if (XDiff > lenThreshold || YDiff > lenThreshold || ZDiff > lenThreshold) {
            return true;
        }

        return false;
    }
    
    bool IsGumMinLen(core::Mesh& mesh, std::vector<int>& groupFace, double minLen = 3.0) {
        auto& meshIndex = mesh.GetIndex();
        std::vector<core::Vector3> coord = mesh.GetPoint();

        // judge length
        double XMin = std::numeric_limits<double>::max(), XMax = std::numeric_limits<double>::lowest();
        double YMin = std::numeric_limits<double>::max(), YMax = std::numeric_limits<double>::lowest();
        double ZMin = std::numeric_limits<double>::max(), ZMax = std::numeric_limits<double>::lowest();

        std::vector<core::Vector3> points;
        bool isHoleBoundary = false;

        int nHole = 0;
        for (int fId : groupFace) {
            core::Vector3 faceOnePt = coord[meshIndex[3 * fId]];
            double ptX = faceOnePt.x(), ptY = faceOnePt.y(), ptZ = faceOnePt.z();
            XMax = ptX > XMax ? ptX : XMax;
            XMin = ptX < XMin ? ptX : XMin;
            YMax = ptY > YMax ? ptY : YMax;
            YMin = ptY < YMin ? ptY : YMin;
            ZMax = ptZ > ZMax ? ptZ : ZMax;
            ZMin = ptZ < ZMin ? ptZ : ZMin;
        }
        double XDiff = XMax - XMin;
        double YDiff = YMax - YMin;
        double ZDiff = ZMax - ZMin;

        if (XDiff < minLen && YDiff < minLen && ZDiff < minLen) {
            return true;
        }

        return false;
    }

    bool IsGumMaxMinLen(core::Mesh& mesh, std::vector<int>& groupFace, double maxLen = 15.0, double minLen = 3.0) {
        auto& meshIndex = mesh.GetIndex();
        std::vector<core::Vector3> coord = mesh.GetPoint();

        // judge length
        double XMin = std::numeric_limits<double>::max(), XMax = std::numeric_limits<double>::lowest();
        double YMin = std::numeric_limits<double>::max(), YMax = std::numeric_limits<double>::lowest();
        double ZMin = std::numeric_limits<double>::max(), ZMax = std::numeric_limits<double>::lowest();

        std::vector<core::Vector3> points;
        bool isHoleBoundary = false;

        int nHole = 0;
        for (int fId : groupFace) {
            core::Vector3 faceOnePt = coord[meshIndex[3 * fId]];
            double ptX = faceOnePt.x(), ptY = faceOnePt.y(), ptZ = faceOnePt.z();
            XMax = ptX > XMax ? ptX : XMax;
            XMin = ptX < XMin ? ptX : XMin;
            YMax = ptY > YMax ? ptY : YMax;
            YMin = ptY < YMin ? ptY : YMin;
            ZMax = ptZ > ZMax ? ptZ : ZMax;
            ZMin = ptZ < ZMin ? ptZ : ZMin;
        }
        double XDiff = XMax - XMin;
        double YDiff = YMax - YMin;
        double ZDiff = ZMax - ZMin;

        if (XDiff > maxLen || YDiff > maxLen || ZDiff > maxLen) {
            return true;
        }

        if (XDiff < minLen && YDiff < minLen && ZDiff < minLen) {
            return true;
        }

        return false;
    }

    void FilterByMinCruv(core::Mesh& mesh, std::vector<double> & curv1, std::vector<double>& curv2, std::vector<bool>& visMark, std::vector<int>& faceFlag, std::vector<std::vector<int> >& groupFaces, double minCurvThreshold = -0.5, int nSmall = 50) {
        const int nFaces = mesh.GetNumFace();
        auto& meshIndex = mesh.GetIndex();

        for (int fId = 0; fId < nFaces; ++fId)
        {
            // find
            if (visMark[fId]) { continue; }
            visMark[fId] = true;

            bool isMinC = false;
            for (int k = 0; k < 3; ++k) {
                double minC = std::min(curv1[meshIndex[3 * fId + k]], curv2[meshIndex[3 * fId + k]]);
                if (minC < minCurvThreshold) {
                    isMinC = true;
                    break;
                }
            }

            if (!isMinC) { continue; }

            // group
            std::vector<int> tempFaceGroup;
            std::queue<int> tempFaceQueue;
            tempFaceQueue.push(fId);
            tempFaceGroup.push_back(fId);
            std::vector<int> neighbors;
            while (!tempFaceQueue.empty()) {
                int curFace = tempFaceQueue.front();
                tempFaceQueue.pop();
                mesh.GetNeighborFace(curFace, neighbors);
                for (int j = 0; j < neighbors.size(); ++j) {
                    int face = neighbors[j];
                    if (visMark[face]) {
                        continue;
                    }
                    visMark[face] = true;

                    isMinC = false;
                    for (int k = 0; k < 3; ++k) {
                        double minC = std::min(curv1[meshIndex[3 * face + k]], curv2[meshIndex[3 * face + k]]);
                        if (minC < minCurvThreshold) {
                            isMinC = true;
                            break;
                        }
                    }

                    if (isMinC) {
                        tempFaceGroup.push_back(face);
                        tempFaceQueue.push(face);
                    }

                }
            }
            if (tempFaceGroup.size() < nSmall) { continue; }   // filter small 
            for (int f : tempFaceGroup) {
                faceFlag[f] = int(groupFaces.size());
            }
            groupFaces.push_back(tempFaceGroup);
        }

        return;
    }

    void SegByMaxCruv(core::Mesh& mesh, std::vector<double>& curv1, std::vector<double>& curv2, std::vector<int>& groupFaceNeedSeg, std::vector<bool>& visMark, std::vector<int>& faceFlag, std::vector<std::vector<int> >& groupMaxCruvFaces, double maxCurvThreshold = 0.5, int nSmall = 50) {
        const int nFaces = mesh.GetNumFace();
        auto& meshIndex = mesh.GetIndex();

        for (int fId : groupFaceNeedSeg) {
            // find 
            if (visMark[fId]) { continue; }
            visMark[fId] = true;

            int flag = faceFlag[fId];
            bool isMaxC = true;
            for (int k = 0; k < 3; ++k) {
                double maxC = std::max(curv1[meshIndex[3 * fId + k]], curv2[meshIndex[3 * fId + k]]);
                if (maxC < maxCurvThreshold) {
                    isMaxC = false;
                    break;
                }
            }

            if (isMaxC) { continue; }

            // group
            std::vector<int> tempFaceGroup;
            std::queue<int> tempFaceQueue;
            tempFaceQueue.push(fId);
            tempFaceGroup.push_back(fId);
            std::vector<int> neighbors;
            while (!tempFaceQueue.empty()) {
                int curFace = tempFaceQueue.front();
                tempFaceQueue.pop();
                mesh.GetNeighborFace(curFace, neighbors);
                for (int j = 0; j < neighbors.size(); ++j) {
                    int face = neighbors[j];
                    if (visMark[face]) {
                        continue;
                    }
                    visMark[face] = true;

                    if (faceFlag[face] != flag) { continue; }

                    isMaxC = true;
                    for (int k = 0; k < 3; ++k) {

                        double maxC = std::max(curv1[meshIndex[3 * face + k]], curv2[meshIndex[3 * face + k]]);
                        if (maxC < maxCurvThreshold) {
                            isMaxC = false;
                            break;
                        }
                    }

                    if (!isMaxC) {
                        tempFaceGroup.push_back(face);
                        tempFaceQueue.push(face);
                    }

                }
            }

            // filter small
            if (tempFaceGroup.size() < nSmall) { 
                if (tempFaceGroup.size() < 10) {
                    for (int f : tempFaceGroup) {
                        faceFlag[f] = -1;
                    }
                }
                continue; 
            }    

            groupMaxCruvFaces.push_back(tempFaceGroup);
            
        }
        return;
    }

    void SegByNormal(core::Mesh& mesh, std::vector<int>& groupFaceNeedSeg, std::vector<bool>& visMark, std::vector<int>& faceFlag, std::vector<std::vector<int> >& groupFaces, 
        std::vector<core::Vector3>& groupNormals, double normalThreshold = 0.3, int nSmall = 50) {
        const int nFaces = mesh.GetNumFace();
        auto& meshIndex = mesh.GetIndex();
        auto& meshFaceNormalArray = mesh.GetFaceNormal();

        for (int fId : groupFaceNeedSeg) {
            // find 
            if (visMark[fId]) { continue; }
            visMark[fId] = true;

            int flag = faceFlag[fId];
            if (flag == -1) { continue; }
            std::vector<core::Vector3> normals;
            core::Vector3 refNormal = meshFaceNormalArray[fId];
            normals.push_back(refNormal);

            // group
            std::vector<int> tempFaceGroup;
            std::queue<int> tempFaceQueue;
            tempFaceQueue.push(fId);
            tempFaceGroup.push_back(fId);
            std::vector<int> neighbors;
            while (!tempFaceQueue.empty()) {
                int curFace = tempFaceQueue.front();
                tempFaceQueue.pop();
                mesh.GetNeighborFace(curFace, neighbors);
                for (int j = 0; j < neighbors.size(); ++j) {
                    int face = neighbors[j];
                    if (visMark[face]) {
                        continue;
                    }
                    visMark[face] = true;
                    core::Vector3 faceNormal = meshFaceNormalArray[face];
                    if (faceFlag[face] != flag || faceNormal.Dot(refNormal) < normalThreshold) { continue; }

                    refNormal += faceNormal;
                    normals.push_back(faceNormal);
                    refNormal.Normalize();
                    tempFaceGroup.push_back(face);
                    tempFaceQueue.push(face);
                }
            }

            // filter small
            if (tempFaceGroup.size() < nSmall) {
                for (int f : tempFaceGroup) {
                    faceFlag[f] = -1;
                }
                continue;
            }

            // compute main dir
            core::Vector3 priDir, secDir;
            core::PrincipalAxis::Compute(normals, priDir, secDir);

            double dotValue = 0.0;
            for (core::Vector3 normal : normals) {
                dotValue += normal.Dot(priDir);
            }
            if (dotValue < 0) {
                priDir.Negate();
            }

            // filter plane
            bool isPlane = true;
            for (core::Vector3 normal : normals) {
                if (normal.Dot(priDir) < 0.9995) {
                    isPlane = false;
                    break;
                }
            }
            if (isPlane) {
                for (int f : tempFaceGroup) {
                    faceFlag[f] = -1;
                }
                continue;
            }

            groupFaces.push_back(tempFaceGroup);
            groupNormals.push_back(priDir);
            //groupNormals.push_back(refNormal);
        }
        return;
    }

    void GroupByFaceFlag(core::Mesh& mesh, std::vector<double>& curv1, std::vector<double>& curv2, std::vector<bool>& visMarkGroup, std::vector<int>& groupFaceNeedSeg, std::vector<int>& faceFlag,
        std::vector<std::vector<int> >& groupFaces, int nSmall = 50, double minLen = 3.5)
    {
        const int nFaces = mesh.GetNumFace();
        auto& meshIndex = mesh.GetIndex();

        for (int fId : groupFaceNeedSeg) {
            // find 
            if (faceFlag[fId] == -1 || visMarkGroup[fId]) { continue; }
            int flag = faceFlag[fId];

            // group
            std::vector<int> tempFaceGroup;
            std::queue<int> tempFaceQueue;
            tempFaceQueue.push(fId);
            tempFaceGroup.push_back(fId);
            visMarkGroup[fId] = true;

            std::vector<int> neighbors;
            while (!tempFaceQueue.empty()) {
                int curFace = tempFaceQueue.front();
                tempFaceQueue.pop();
                mesh.GetNeighborFace(curFace, neighbors);
                for (int j = 0; j < neighbors.size(); ++j) {
                    int face = neighbors[j];
                    if (visMarkGroup[face] || faceFlag[face] != flag) {
                        continue; 
                    }

                    tempFaceGroup.push_back(face);
                    tempFaceQueue.push(face);
                    visMarkGroup[face] = true;
                }
            }

            if (tempFaceGroup.size() < nSmall) { continue; }   // filter small

            if (!IsGumMinLen(mesh, tempFaceGroup, minLen)) {
                groupFaces.push_back(tempFaceGroup);
            }
        }
        return;
    }
    
    void FindSharpEdgesAndGroup(core::Mesh& mesh, std::vector<size_t>& edgeCandidateArray, std::vector<std::vector<int> >& groupEdgeArray, double dotThreshold = 0.05) {
        mesh.GenerateEdgeAndAdjacency();
        mesh.GenerateFaceNormal(false, true);
        auto& coord = mesh.GetPoint();
        auto& index = mesh.GetIndex();
        auto& faceNormal = mesh.GetFaceNormal();
        auto& faceArea = mesh.GetFaceArea();

        // find sharp edge
        auto& edgeArray = mesh.GetEdgeArray();
        auto& edgeFaceArray = mesh.GetEdgeFaceAdjacency();
        auto& pointEdgeArray = mesh.GetVertexEdgeAdjacency();

        std::vector<bool> edgeMark(edgeArray.size(), false);
        std::vector<bool> pointSharpMark(coord.size(), false);

        for (size_t i = 0; i < edgeArray.size(); ++i)
        {
            auto& ef = edgeFaceArray[i];
            if (ef.size() >= 2)
            {
                double d = faceNormal[ef[0]].Dot(faceNormal[ef[1]]);
                if (std::abs(d) <= dotThreshold)
                {
                    edgeMark[i] = true;
                    edgeCandidateArray.push_back(i);
                }
            }
        }

        // group sharp edge
        {
            groupEdgeArray.clear();
            size_t currentEdgeSearchIndex = 0;
            std::queue<int> edgeQueue;
            std::vector<int> currentEdgeArray;
            std::vector<int> groupEdge;
            while (true)
            {
                int startEdge = -1;
                for (size_t i = currentEdgeSearchIndex; i < edgeCandidateArray.size(); ++i)
                {
                    int e = edgeCandidateArray[i];
                    groupEdge.clear();
                    if (edgeMark[e])
                    {
                        startEdge = e;
                        currentEdgeSearchIndex = i + 1;
                        edgeMark[e] = false;
                        groupEdge.push_back(e);
                        break;
                    }
                }
                if (startEdge == -1) {
                    break;
                }

                int v0 = edgeArray[startEdge].start_, v1 = edgeArray[startEdge].end_;
                edgeQueue.push(startEdge);

                while (!edgeQueue.empty())
                {
                    int curEdge = edgeQueue.front();
                    edgeQueue.pop();

                    v0 = edgeArray[curEdge].start_;
                    v1 = edgeArray[curEdge].end_;
                    if (!pointSharpMark[v0])
                    {
                        for (auto e : pointEdgeArray[v0])
                        {
                            if (edgeMark[e])
                            {
                                groupEdge.push_back(e);
                                edgeQueue.push(e);
                                edgeMark[e] = false;
                            }
                        }
                        pointSharpMark[v0] = true;
                    }

                    if (!pointSharpMark[v1])
                    {
                        for (auto e : pointEdgeArray[v1])
                        {
                            if (edgeMark[e])
                            {
                                groupEdge.push_back(e);
                                edgeQueue.push(e);
                                edgeMark[e] = false;
                            }
                        }
                        pointSharpMark[v1] = true;
                    }
                }
                groupEdgeArray.push_back(groupEdge);
            }
            
        }
    }

    void FindSharpEdgeNBFaces(core::Mesh& mesh, std::vector<std::vector<int> >& groupEdgeArray, std::vector<std::vector<int> >& sharpFaces, std::vector<core::Vector3>& faceSetDirs, double dotThreshold = 0.05, double nbDotThreshold = 0.90) {
        mesh.GenerateEdgeAndAdjacency();
        mesh.GenerateFaceNormal(false, true);
        auto& coord = mesh.GetPoint();
        auto& index = mesh.GetIndex();
        auto& faceNormal = mesh.GetFaceNormal();
        auto& faceArea = mesh.GetFaceArea();
        auto& edgeFaceArray = mesh.GetEdgeFaceAdjacency();

        std::vector<bool> faceMark(mesh.GetNumFace(), false);
        for (auto groupEdge : groupEdgeArray) {
            for (int e : groupEdge) {
                auto& ef = edgeFaceArray[e];
                if (ef.size() < 2) { continue; }
                int f1 = ef[0];
                int f2 = ef[1];
                core::Vector3 f1Normal = faceNormal[f1];
                core::Vector3 f2Normal = faceNormal[f2];

                std::queue<int> edgeQueue1, edgeQueue2;
                std::vector<int> f1Region, f2Region;

                if (!faceMark[f1]) {
                    edgeQueue1.push(f1);
                    faceMark[f1] = true;
                    f1Region.push_back(f1);
                }
                if (!faceMark[f2]) {
                    edgeQueue2.push(f2);
                    faceMark[f2] = true;
                    f2Region.push_back(f2);
                }

                while (!edgeQueue1.empty()) {
                    int curF = edgeQueue1.front();
                    edgeQueue1.pop();
                    std::vector<int> nbFaces;
                    mesh.GetNeighborFace(curF, nbFaces);
                    /*mesh.GetFaceStar(curF, nbFaces);*/
                    core::Vector3 curNormal = faceNormal[curF];

                    for (int nbF : nbFaces) {
                        if (faceMark[nbF]) { continue; }
                        core::Vector3 nbNormal = faceNormal[nbF];
                        if (std::abs(f2Normal.Dot(nbNormal)) < dotThreshold && curNormal.Dot(nbNormal) > nbDotThreshold) {
                            edgeQueue1.push(nbF);
                            faceMark[nbF] = true;
                            f1Region.push_back(nbF);
                        }
                    }

                }

                while (!edgeQueue2.empty()) {
                    int curF = edgeQueue2.front();
                    edgeQueue2.pop();
                    std::vector<int> nbFaces;
                    mesh.GetNeighborFace(curF, nbFaces);
                    /*mesh.GetFaceStar(curF, nbFaces);*/
                    core::Vector3 curNormal = faceNormal[curF];

                    for (int nbF : nbFaces) {
                        if (faceMark[nbF]) { continue; }
                        core::Vector3 nbNormal = faceNormal[nbF];
                        if (std::abs(f1Normal.Dot(nbNormal)) < dotThreshold && curNormal.Dot(nbNormal) > nbDotThreshold) {
                            edgeQueue2.push(nbF);
                            faceMark[nbF] = true;
                            f2Region.push_back(nbF);
                        }
                    }
                }

                if (f1Region.size() > 0) {
                    sharpFaces.push_back(f1Region);
                    faceSetDirs.push_back(f2Normal);
                }

                if (f2Region.size() > 0) {
                    sharpFaces.push_back(f2Region);
                    faceSetDirs.push_back(f1Normal);
                }
            }
        }
    }

    double GetDistToConvexHull(core::Mesh& mesh, core::KDTreeSAH& kdtree, std::vector<int> groupFaces, core::Vector3& axis) {
        double dist = 0.0;

        auto& meshIndex = mesh.GetIndex();
        std::vector<core::Vector3> coord = mesh.GetPoint();
        core::Vector3 center;
        for (int f : groupFaces) {
            center += coord[meshIndex[3 * f]];  // just add one
        }
        center = center / groupFaces.size();
        int faceId;
        kdtree.Intersect(center, axis, dist, faceId);
        return dist;
    }

    DentalTexture::DentalTexture()
    {

    }

    DentalTexture::~DentalTexture()
    {

    }
    
    bool DentalTexture::GetToothTexture(core::Mesh& inputMesh, const std::string onnxModelPath, std::vector<std::vector<int> >& groupFaces)
    {
        Mesh mesh;
        mesh.MinimalCopy(inputMesh);
        //core::MeshHealing::RemoveSmallerIslands(mesh, core::MeshHealing::MeshHealingMetric::VOLUME);

        // compute curvature
        std::vector<double> curv1, curv2;
        std::vector<core::Vector3> pdir1, pdir2;
        core::MeshCurvature::ComputeCurvatureTensor(mesh, curv1, curv2, pdir1, pdir2);

        // get mesh info
        mesh.GenerateEdgeAndAdjacency();
        mesh.GenerateFaceNormal();
        const int nVertex = mesh.GetNumPoint();
        const int nFaces = mesh.GetNumFace();
        auto& meshIndex = mesh.GetIndex();
        std::vector<core::Vector3> coord = mesh.GetPoint();
        auto& meshFaceNormalArray = mesh.GetFaceNormal();

        auto& edgeArray = mesh.GetEdgeArray();
        auto& edgeFaceArray = mesh.GetEdgeFaceAdjacency();
        auto& pointEdgeArray = mesh.GetVertexEdgeAdjacency();
        std::vector<int> faceFlag(nFaces, -1);

        std::vector<bool> visMark(nFaces, false);
        // remove sharp face 
        std::vector<size_t> edgeCandidateArray;
        std::vector<std::vector<int> > groupEdgeArray;
        FindSharpEdgesAndGroup(mesh, edgeCandidateArray, groupEdgeArray);
        if (edgeCandidateArray.size() > 1) {     
            // find sharp edge neighbor face
            std::vector<std::vector<int> > sharpFaces;
            std::vector<core::Vector3> faceSetDirs;
            FindSharpEdgeNBFaces(mesh, groupEdgeArray, sharpFaces, faceSetDirs);
            for (std::vector<int> sharpFace : sharpFaces) {
                for (int f : sharpFace) {
                    visMark[f] = true;
                }
            }
        }

        double maxCurvThreshold = 0.50;
        double minCurvThreshold = -0.55;
        double normalThreshold = 0.30;
        int nSmall = 50;
        int nLarge = 400;
        int nSelect = 30;
        double maxLen = 15.0;  // teeth size
        double minLen = 3.5;

        // 1.find all initial candidate points by min Curv
        FilterByMinCruv(mesh, curv1, curv2, visMark, faceFlag, groupFaces, minCurvThreshold, nSmall);

        if (groupFaces.size() < 1) { return false; };
        std::sort(groupFaces.begin(), groupFaces.end(), [](const std::vector<int>& a, const std::vector<int>& b) {return a.size() > b.size(); });
        // fliter some small
        int removeId = std::min(int(groupFaces.size()), nSelect);
        groupFaces.erase(groupFaces.begin() + removeId, groupFaces.end());

# ifdef _DEBUG
        if (false) {
            core::STLFileHandler stlHandler;
            stlHandler.Write(L"D:/" + inputMesh.GetName() + L".stl", mesh);

            std::vector<int> potentialFaceIDs;
            for (std::vector<int> groupFace : groupFaces) {
                for (int f : groupFace) {
                    potentialFaceIDs.push_back(f);
                }
            }
            core::Mesh potentialMesh;
            potentialMesh.MinimalCopy(mesh);
            std::vector<int> newIndex;
            potentialMesh.ExpandFaceIdIntoIndex(potentialFaceIDs, newIndex);
            auto& oldIndex = potentialMesh.GetIndex();
            oldIndex.swap(newIndex);
            potentialMesh.RemoveUnreferencedPoint();
            stlHandler.Write(L"D:/" + inputMesh.GetName() + L"_potential.stl", potentialMesh);

            int n = 1;
            for (std::vector<int> groupFace : groupFaces) {
                if (groupFace.size() < 400) {
                    continue;
                }
                core::Mesh potentialMeshI;
                potentialMeshI.MinimalCopy(mesh);
                std::vector<int> newIndexI;
                potentialMeshI.ExpandFaceIdIntoIndex(groupFace, newIndexI);
                auto& oldIndexI = potentialMeshI.GetIndex();
                oldIndexI.swap(newIndexI);
                potentialMeshI.RemoveUnreferencedPoint();
                stlHandler.Write(L"D:/" + inputMesh.GetName() + L"_potential_" + std::to_wstring(n) + L".stl", potentialMeshI);
                n += 1;
            }

        }
# endif

        // 2.seg largest by max cruv
        std::vector<std::vector<int> > newGroupFaces;
        std::vector<bool> visMarkMaxCruv(nFaces, false);
        for (std::vector<int> groupFace : groupFaces) {
            if (groupFace.size() > nLarge) {
                std::vector<std::vector<int> > groupMaxCruvFaces;
                SegByMaxCruv(mesh, curv1, curv2, groupFace, visMarkMaxCruv, faceFlag, groupMaxCruvFaces, maxCurvThreshold, nSmall);

                for (std::vector<int> groupMCFace : groupMaxCruvFaces) {
                    // judge length , may be need to split by other cluster method
                    bool isGumLine = IsGumMaxLen(mesh, groupMCFace, maxLen);  
                    if (isGumLine) {
                        /*for (int f : groupMCFace) {           // drop, it can be faster
                            faceFlag[f] = -1;
                        }*/

                        newGroupFaces.push_back(groupMCFace);   // may include texture
                        int flag = groupMCFace.size();
                        for (int f : groupMCFace) {
                            faceFlag[f] = flag;
                        }
                    }
                }
            }
            else {
                // judge length
                bool isGumLine = IsGumMaxLen(mesh, groupFace, maxLen);
                if (isGumLine) {
                    /*for (int f : groupFace) {
                        faceFlag[f] = -1;
                    }*/

                    newGroupFaces.push_back(groupFace);
                    int flag = groupFace.size();
                    for (int f : groupFace) {
                        faceFlag[f] = flag;
                    }
                }
            }
        }

        // group again
        //std::vector<std::vector<int> > newGroupFaces;
        std::vector<bool> visMarkGroup(nFaces, false);
        for (std::vector<int> groupFace : groupFaces) {
            if (groupFace.size() > nLarge) {
                GroupByFaceFlag(mesh, curv1, curv2, visMarkGroup, groupFace, faceFlag, newGroupFaces, nSmall, minLen);
            }
            else {
                if (groupFace.size() > 0 && faceFlag[groupFace[0]] != -1 && !IsGumMinLen(mesh, groupFace, minLen)) {
                    newGroupFaces.push_back(groupFace);
                }
            }
        }
        groupFaces.swap(newGroupFaces);

        /*if (groupFaces.size() < 1) { return false; };
        std::sort(groupFaces.begin(), groupFaces.end(), [](const std::vector<int>& a, const std::vector<int>& b) {return a.size() > b.size(); });*/

        // convex hull
        double refDist = 8.0;
        core::Mesh hullMesh;
        MeshConvexHull::ComputeConvexHull(mesh, hullMesh);
        core::KDTreeSAH kdtree;
        kdtree.Build(hullMesh);

        // 3.seg largest by normal
        newGroupFaces.clear();
        std::vector<core::Vector3> groupNormals;
        std::vector<bool> visMarkNormal(nFaces, false);
        for (std::vector<int> groupFace : groupFaces) {
            std::vector<std::vector<int> > groupNormalFaces;
            std::vector<core::Vector3> groupFaceNormals;
            SegByNormal(mesh, groupFace, visMarkNormal, faceFlag, groupNormalFaces, groupFaceNormals, normalThreshold, nSmall);
            // filter
            for (int i = 0; i < groupNormalFaces.size(); ++i) {
                std::vector<int> groupNMFace = groupNormalFaces[i];
                // judge length
                if (IsGumMinLen(mesh, groupNMFace, minLen)) {   //IsGumMaxMinLen(mesh, groupNMFace, maxLen, minLen)
                    for (int f : groupNMFace) {
                        faceFlag[f] = -1;
                    }
                }
                else {
                    // filter by dist to convex hull 
                    double dist = GetDistToConvexHull(mesh, kdtree, groupNMFace, groupFaceNormals[i]);
                    if (dist > refDist) { continue; }

                    // TODO: may be need seg IsGumMaxLen by dist
                    newGroupFaces.push_back(groupNMFace);
                    groupNormals.push_back(groupFaceNormals[i]);
                }
            }
        }

        // 4.project and classify by AI
        if (!Rasterizer::Instance().IsValid())
        {
            if (!Rasterizer::Instance().Init())
            {
                HGAPI_LOG(DentalTexture::GetToothBaseTexture, L"rasterizer init failed");
                return false;
            }
            HGAPI_LOG_MESSAGE(L"rasterizer inited in GumLineExtraction");
        }

        // load AI classfiy model
        ClassifyByAI clsByAI;
        if (!clsByAI.LoadModel(onnxModelPath)) {
            HGAPI_LOG(DentalTexture::GetToothBaseTexture, L"Load Model failed");
            return false;
        }

        // filter by AI
        groupFaces.clear();
        for (int i = 0; i < newGroupFaces.size(); ++i) {
            core::Image image;
            core::Matrix4 mat = core::Matrix4::Identity();
            mat = core::Matrix4::Rotation(groupNormals[i], core::Vector3(0.0, 0.0, 1.0));
            Rasterizer::Instance().GenerateProjectedImage(mesh, 0.1, image, nullptr, std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max(), mat, &newGroupFaces[i]);
                
            // predict
            int classId = -1;   // 0: other,  1: texture
            if (clsByAI.Predict(image, classId) && classId == 1) {
                groupFaces.push_back(newGroupFaces[i]);
            }
        }

# ifdef _DEBUG
        if (false) {
            core::STLFileHandler stlHandler;

            // save all
            std::vector<int> potentialFaceIDs;
            for (std::vector<int> groupFace : newGroupFaces) {
                for (int f : groupFace) {
                    potentialFaceIDs.push_back(f);
                }
            }
            core::Mesh potentialMesh;
            potentialMesh.MinimalCopy(inputMesh);
            std::vector<int> newIndex;
            potentialMesh.ExpandFaceIdIntoIndex(potentialFaceIDs, newIndex);
            auto& oldIndex = potentialMesh.GetIndex();
            oldIndex.swap(newIndex);
            potentialMesh.RemoveUnreferencedPoint();
            stlHandler.Write(L"D:/" + inputMesh.GetName() + L"_potential_last.stl", potentialMesh);

            potentialFaceIDs.clear();
            for (std::vector<int> groupFace : groupFaces) {
                for (int f : groupFace) {
                    potentialFaceIDs.push_back(f);
                }
            }
            core::Mesh potentialMesh1;
            potentialMesh1.MinimalCopy(inputMesh);
            std::vector<int> newIndex1;
            potentialMesh1.ExpandFaceIdIntoIndex(potentialFaceIDs, newIndex1);
            auto& oldIndex1 = potentialMesh1.GetIndex();
            oldIndex1.swap(newIndex1);
            potentialMesh1.RemoveUnreferencedPoint();
            stlHandler.Write(L"D:/" + inputMesh.GetName() + L"_potential_last_filterByAI.stl", potentialMesh1);

        }
# endif  
        
        return true;
    }

    bool DentalTexture::GetMinCruvPts(core::Mesh& inputMesh, std::vector<core::Vector3>& minCurvPts, std::vector<core::Vector3>& ptNormals, double minCruvThreshold)
    {
        Mesh mesh;
        mesh.MinimalCopy(inputMesh);
        core::MeshHealing::RemoveSmallerIslands(mesh, core::MeshHealing::MeshHealingMetric::VOLUME);
        core::MeshHealing::RemoveDanglingFace(mesh);

        // compute curvature
        std::vector<double> curv1, curv2;
        std::vector<core::Vector3> pdir1, pdir2;
        core::MeshCurvature::ComputeCurvatureTensor(mesh, curv1, curv2, pdir1, pdir2);

        // get mesh info
        mesh.GenerateEdgeAndAdjacency();
        mesh.GenerateFaceNormal();
        const int nVertex = mesh.GetNumPoint();
        const int nFaces = mesh.GetNumFace();
        auto& meshIndex = mesh.GetIndex();
        std::vector<core::Vector3> coord = mesh.GetPoint();
        auto& meshVertexNormalArray = mesh.GetVertexNormal();

        auto& edgeArray = mesh.GetEdgeArray();
        auto& edgeFaceArray = mesh.GetEdgeFaceAdjacency();
        auto& pointEdgeArray = mesh.GetVertexEdgeAdjacency();

        std::vector<bool> visMark(nVertex, false);
        // remove sharp face 
        std::vector<size_t> edgeCandidateArray;
        std::vector<std::vector<int> > groupEdgeArray;
        FindSharpEdgesAndGroup(mesh, edgeCandidateArray, groupEdgeArray);
        if (edgeCandidateArray.size() > 1) {
            // find sharp edge neighbor face
            std::vector<std::vector<int> > sharpFaces;
            std::vector<core::Vector3> faceSetDirs;
            FindSharpEdgeNBFaces(mesh, groupEdgeArray, sharpFaces, faceSetDirs);
            for (std::vector<int> sharpFace : sharpFaces) {
                for (int f : sharpFace) {
                    for (int k = 0; k < 3; ++k) {
                        visMark[meshIndex[3 * f + k]] = true;
                    }
                }
            }
        }

        // compute convex hull
        double refDist = 10.0;
        core::Mesh hullMesh;
        MeshConvexHull::ComputeConvexHull(mesh, hullMesh);
        core::KDTreeSAH kdtree;
        kdtree.Build(hullMesh);

        // find minCruv points  // May be need to group faces and remove small group
        for (int v = 0; v < nVertex; ++v) {
            if (visMark[v]) { continue; }

            visMark[v] = true;
            double minC = std::min(curv1[v], curv2[v]);
            if (minC > minCruvThreshold) {
                continue;
            }

            std::vector<int> neighborVerteses;
            mesh.GetNeighborVertex(v, neighborVerteses);   // can use KN, iter k times
            double dist = 0.0;
            int faceId;
            kdtree.Intersect(coord[v], meshVertexNormalArray[v], dist, faceId);
            if (dist > refDist) {
                for (int nv : neighborVerteses) {
                    visMark[nv] = true;
                }
            }
            else {
                int n = 0;
                for (int nv : neighborVerteses) {
                    if (visMark[nv]) { continue; }

                    double minCNV = std::min(curv1[nv], curv2[nv]);
                    if (minCNV < minCruvThreshold) {
                        minCurvPts.push_back(coord[nv]);
                        ptNormals.push_back(meshVertexNormalArray[nv]);
                        n += 1;
                    }

                    visMark[nv] = true;
                }
                if (n > 0) { 
                    minCurvPts.push_back(coord[v]);     // can remove isolated point
                    ptNormals.push_back(meshVertexNormalArray[v]);
                }
                
            }
        }

#ifdef _DEBUG
        if (false) {
            core::PTSHandler ptsHandler;
            ptsHandler.Export(L"D:/" + inputMesh.GetName() + L".pts", minCurvPts);
        }
#endif

        return true;
    }

    bool GeneratePtIdxByAI(const char* meshInfo, std::vector<int>& labels, const std::string IP, const std::string Port, std::string route, const  DentalTexture::ReqMode reqMode) {
        core::MeshCNNHTTP meshCNNHTTP = core::MeshCNNHTTP(IP, Port);

        // used to save predict result
        std::string predictRes;

        // sent request
        if (reqMode == DentalTexture::ReqMode::UPLOAD_FILE) {
            meshCNNHTTP.reqMeshCNNFileUpload(meshInfo, predictRes, route);   // meshInfo is file path and filename
        }
        else {
            meshCNNHTTP.reqMeshCNNObjJson(meshInfo, predictRes, route);      // meshInfo is json data
        }

        // parse result
        core::MeshCNNHTTP::stringReplace(predictRes, "\\\"", "\"");
        predictRes = std::string(predictRes, 1, predictRes.size() - 2);
        Json::CharReaderBuilder readerBuild;
        Json::CharReader* reader(readerBuild.newCharReader());
        Json::Value rcvRes;
        JSONCPP_STRING jsonErrs;
        bool parseOK = reader->parse(predictRes.c_str(), predictRes.c_str() + predictRes.size(), &rcvRes, &jsonErrs);

        delete reader;
        if (!parseOK) {
            //std::cout << "Failed to parse the rcvRes!" << std::endl;
            return false;
        }

        std::string returnCode = rcvRes["returnCode"].asString();  // "Successed!" or "Failed!"
        if (returnCode != "Successed!") {
            //std::cout << "Predict is Failed" << std::endl;
            return false;
        }

        // get labels
        labels.clear();
        Json::Value rcvResItem;
        int rcvSize = rcvRes["text"].size();
        for (int i = 0; i < rcvSize; ++i) {
            rcvResItem = rcvRes["text"][i];
            int label = rcvResItem.asInt();
            labels.push_back(label);
        }
        return true;
    }

    bool DentalTexture::GetToothTextureHTTP(core::Mesh& mesh, std::vector<core::Vector3>& texturePts, std::vector<core::Vector3>& ptsNormals, const std::string IP, const std::string Port, bool useNormal, std::string route, const ReqMode reqMode, double minCruvThreshold) {
        // get candidate points and normals
        std::vector<core::Vector3> candidatePts;
        std::vector<core::Vector3> candidatePtNormals;
        if (!GetMinCruvPts(mesh, candidatePts, candidatePtNormals, minCruvThreshold)) {
            return false;
        }

        // get inputs feature
        std::stringstream ss;
        for (size_t i = 0; i < candidatePts.size(); ++i)
        {
            core::Vector3 pt = candidatePts[i];
            if (useNormal) {
                core::Vector3 normal = candidatePtNormals[i];
                ss << pt.x() << " " << pt.y() << " " << pt.z() << " " << normal.x() << " " << normal.y() << " " << normal.z() << "\n";
            }
            else {
                ss << pt.x() << " " << pt.y() << " " << pt.z() << "\n";
            }
        }

        // get texture pts ids by AI
        std::vector<int> idxs;
        if (reqMode == ReqMode::JSON_DATA) {
            // request
            if (!GeneratePtIdxByAI(ss.str().c_str(), idxs, IP, Port, route, reqMode)) {
                return false;
            }
        }
        else {
            // save to txt file
            std::string txtSavePath = "./_inputs.txt";
            std::fstream destFile(txtSavePath, std::ios::out);
            destFile << ss.str();
            destFile.close();

            // convert save path to char *
            const char* txtPath = txtSavePath.c_str();

            if (!GeneratePtIdxByAI(txtPath, idxs, IP, Port, route, reqMode)) {
                return false;
            }

        }

        // result process
        for (int i = 0; i < idxs.size(); ++i) {
            int idx = idxs[i];
            texturePts.push_back(candidatePts[idx]);
            ptsNormals.push_back(candidatePtNormals[idx]);
        }
        // TODO: remove some noise points
        // tip1: rotate point by PCA normal direction, then calc the avg Z and remove the point that z value is outer of 2-std

         
#ifdef _DEBUG
        if (false) {
            core::PTSHandler ptsHandler;
            ptsHandler.Export(L"D:/" + mesh.GetName() + L"_texture.pts", texturePts);
        }
#endif
        return true;
    }

}
