#pragma once

#include "mesh/mesh.h" 

namespace dental
{

struct BracketRemoveOpt { 
    std::string AIModelPath;  // AI model 
    bool bHasIronWire;        // whether has iron wire
    bool bHasRing;            // whether has ring

    BracketRemoveOpt()
        : AIModelPath("//10.99.11.210/models/nightGuard/AIModel/Tooth_2D_V2.onnx")
        , bHasIronWire(true)
        , bHasRing(true)
    {}
};

class BracketRemove
{
public:
    BracketRemove(core::Mesh& mesh, const BracketRemoveOpt& opt);
    ~BracketRemove();
   
    bool FindBrackets();  
    core::Mesh GetMesh();

    bool RemoveBracketByFace(const std::vector<int>& oneBracketFaceIds);
    bool RemoveBracketByVertex(const std::vector<int>& oneBracketVertexIds);
    bool RemoveBracket();
    
private:
    BracketRemoveOpt opt_;
    core::Mesh inMesh_; 
    std::vector<int>  allBracketVIds_;
    std::vector<char> vMarks_;
};

}
