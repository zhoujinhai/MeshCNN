
#include "meshCNN.h"

MeshCNN::MeshCNN():pModule(nullptr), pInfDict(nullptr), pInfClass(nullptr), pInfIns(nullptr)
{

}

MeshCNN::~MeshCNN()
{
    // release
    Py_DECREF(pInfClass);
    Py_DECREF(pInfDict);
    Py_DECREF(pModule);
    Py_DECREF(pInfIns);

    if ( Py_FinalizeEx() < 0 ){
        exit(120);
    }
    PyMem_RawFree(program);
}

bool MeshCNN::AIInit(const char* runPath)
{
    program = Py_DecodeLocale(runPath, nullptr);  
    if ( !program ){
        std::cout << "Fatal Error: cannot decode argv[0]: " << runPath << std::endl;
        return false;
    }
    Py_SetProgramName(program);
    Py_SetPythonHome((wchar_t*)L"/home/heygears/anaconda3/envs/MeshCNN/");
    Py_Initialize();
    if ( !Py_IsInitialized() ){
        std::cout << "Python init failed!" << std::endl;
	return false;
    }

    import_array();  // load numpy api
    return true;
}

bool MeshCNN::AILoadModel()
{
    // set run path
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/heygears/jinhai_zhou/work/C++_python/MeshClass')");
    PyRun_SimpleString("sys.path.append('/home/heygears/jinhai_zhou/work/C++_python/MeshClass/mesh_net/')");
    
    // load inference.py
    pModule = PyImport_ImportModule("inference_class");
    if ( !pModule ){
        std::cout << "inference module can not found! " << std::endl;
        return false;
    }
    
    pInfDict = PyModule_GetDict(pModule);

    // load class
    pInfClass = PyDict_GetItemString(pInfDict, "InferenceClass");

    // create instance 
    if ( PyCallable_Check(pInfClass) ){
        pInfIns = PyObject_CallObject(pInfClass, nullptr);
    }
    return true;
}

bool MeshCNN::AIPredict(const char* dentalPath, std::vector<std::vector<float> > &predictRes)
{
    // call func
    PyObject* predictResult = PyObject_CallMethod(pInfIns, "inference", "s", dentalPath);
    if ( !predictResult ){
        return false;
    }

    // get output
    PyObject* resItem = PyList_GetItem(predictResult, 0);
    if ( !resItem ){
         return false;
    }
    
    PyObject* classValue = nullptr;
    PyObject *classItem = nullptr;
    float temp = 0.0;
    int classSize = PyObject_Size(resItem);
    for ( int i = 0; i < classSize; ++i ){
        std::vector<float> classData;
        classData.clear();
        classValue = PyList_GetItem(resItem, i);
        int edgeSize = PyObject_Size(classValue);
	for ( int j = 0; j < edgeSize; ++j ){   
	    classItem = PyList_GetItem(classValue, j);
            PyArg_Parse(classItem, "f", &temp); 
            classData.push_back(temp);
	}
	predictRes.push_back(classData);
    }
    
    Py_DECREF(predictResult);
    Py_DECREF(resItem);
    Py_DECREF(classValue);
    Py_DECREF(classItem);
    
    return true;
}


void printfRes(std::vector<std::vector<float> > &predictRes)
{
     for ( size_t i = 0; i < predictRes.size(); ++i ){
        for ( size_t j = 0; j < predictRes[0].size(); ++j ){
            std::cout << predictRes[i][j] << ",";
        }
        std::cout << "******" << std::endl; 
    }
}


int main(int argc, char *argv[])
{
    if (argc != 2) {
        // usage help
        std::cout << "This code is used to predict dental model's label!" << std::endl;
        std::cout << "Usage: " << argv[0] << " dental_model_path " << std::endl;
        std::cout << "Example: " << argv[0] << " ../test_models/1AQ9W_VS_SET_VSc1_Subsetup6_Maxillar.obj" << std::endl;
        return 1;
    }
    
    // uset to save predict result
    std::vector<std::vector<float> > predictRes;                                      
    
    // start time
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();   
    
    // create instance
    MeshCNN meshCNN;
    // init
    if ( !meshCNN.AIInit(argv[0]) ){
        std::cout << "Init Failed!" << std::endl;
        return -1;
    }
    // load model
    if ( !meshCNN.AILoadModel() ){
        std::cout << "Load AI Model Failed! " << std::endl;
        return -1;
    }
    
    // predict 
    if ( !meshCNN.AIPredict(argv[1], predictRes) ){                           
        std::cout << "AI Predict Failed!" << std::endl;
        return -1;
    }                                    
    
    // end time
    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();  
    // calcu time   
    std::chrono::duration<double> time_span = endTime - startTime;                            
    std::cout << "all code took " << time_span.count() << " seconds." << std::endl;   

    // print res
    // printfRes(predictRes);                                                         
    
    return 0;
}
