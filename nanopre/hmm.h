#ifndef _HMM_H
    #define HMM_H
    #define PY_SSIZE_T_CLEAN
    #include <Python.h>
    #include <numpy/arrayobject.h>
    #include <vector>
#endif
void vaterbi_decoding(PyArrayObject *array, std::vector< std::vector<double> > &decoded, std::vector< std::vector<short> > &path, std::vector<long >& locs);
