#pragma once

#include <python3.6m/Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h> 

#include <iostream>
#include <vector>
#include <chrono>


class MeshCNN
{
public:
        MeshCNN();
        ~MeshCNN();
        bool AIInit(const char* runPath);
        bool AILoadModel();
        bool AIPredict(const char* dentalPath, std::vector<std::vector<float> > &predictRes);
private:
        wchar_t *program;
        PyObject* pModule;
        PyObject* pInfDict;
        PyObject* pInfClass;
        PyObject* pInfIns;
};
