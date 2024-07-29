#include <python3.6m/Python.h>
#include <iostream>


int main(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], nullptr);
    if ( program == nullptr ){
        std::cout << "Fatal Error: cannot decode argv[0]!" << std::endl;
        return -1;
    }
    Py_SetProgramName(program);
    Py_Initialize();
    if ( !Py_IsInitialized() ){
        std::cout << "Python init failed!" << std::endl;
	return 0;
    }
    
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/heygears/work/C++_python/test/test_class')");
    
    // 1. 导入模块及字典
    PyObject* pModule = PyImport_ImportModule("test_class");
    if ( pModule == nullptr ){
        std::cout << "module not found!" << std::endl;
        return 1;
    }

    PyObject* pTestDict = PyModule_GetDict(pModule);
    
    
    // 2. 导入模块中方法或类
    PyObject* pTestClass = PyDict_GetItemString(pTestDict, "Test");

    // 3. 创建实例
    PyObject* pTestInstance = nullptr;
    if ( PyCallable_Check(pTestClass) ){
        pTestInstance = PyObject_CallObject(pTestClass, nullptr);
        std::cout << "---" << std::endl;
    }
    
    // 4. 调用类方法
    PyObject_CallMethod(pTestInstance, "do", nullptr);
    PyObject_CallMethod(pTestInstance, "modify", nullptr);
    PyObject_CallMethod(pTestInstance, "do", nullptr);

    if ( Py_FinalizeEx() < 0 ){
        exit(120);
    }
    PyMem_RawFree(program);
    return 0;
}
