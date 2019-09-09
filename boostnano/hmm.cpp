
#include "hmm.h"
#include <math.h>
#include <limits>
#include <string>
using namespace std;

/** Convert a c++ 2D vector into a numpy array
 *
 * @param const vector< vector<T> >& vec : 2D vector data
 * @return PyArrayObject* array : converted numpy array
 *
 * Transforms an arbitrary 2D C++ vector into a numpy array. Throws in case of
 * unregular shape. The array may contain empty columns or something else, as
 * long as it's shape is square.
 *
 * Warning this routine makes a copy of the memory!
 */
template<typename T>
static PyArrayObject* vector2D_to_nparray(const vector< vector<T> >& vec, int type_num = PyArray_FLOAT){

   // rows not empty
   if( !vec.empty() ){

      // column not empty
      if( !vec[0].empty() ){

        size_t nRows = vec.size();
        size_t nCols = vec[0].size();
        npy_intp dims[2] = {nRows, nCols};
        PyArrayObject* vec_array = (PyArrayObject *) PyArray_SimpleNew(2, dims, type_num);

        T *vec_array_pointer = (T*) PyArray_DATA(vec_array);

        // copy vector line by line ... maybe could be done at one
        for (size_t iRow=0; iRow < vec.size(); ++iRow){

          if( vec[iRow].size() != nCols){
             Py_DECREF(vec_array); // delete
             throw(string("Can not convert vector<vector<T>> to np.array, since c++ matrix shape is not uniform."));
          }

          copy(vec[iRow].begin(),vec[iRow].end(),vec_array_pointer+iRow*nCols);
        }

        return vec_array;

     // Empty columns
     } else {
        npy_intp dims[2] = {vec.size(), 0};
        return (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_FLOAT, 0);
     }


   // no data at all
   } else {
      npy_intp dims[2] = {0, 0};
      return (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_FLOAT, 0);
   }

}


/** Convert a c++ vector into a numpy array
 *
 * @param const vector<T>& vec : 1D vector data
 * @return PyArrayObject* array : converted numpy array
 *
 * Transforms an arbitrary C++ vector into a numpy array. Throws in case of
 * unregular shape. The array may contain empty columns or something else, as
 * long as it's shape is square.
 *
 * Warning this routine makes a copy of the memory!
 */
template<typename T>
static PyArrayObject* vector1D_to_nparray(const vector<T>& vec, int type_num = PyArray_FLOAT){

   // rows not empty
   if( !vec.empty() ){

       size_t nRows = vec.size();
       npy_intp dims[1] = {nRows};

       PyArrayObject* vec_array = (PyArrayObject *) PyArray_SimpleNew(1, dims, type_num);
       T *vec_array_pointer = (T*) PyArray_DATA(vec_array);

       copy(vec.begin(),vec.end(),vec_array_pointer);
       return vec_array;

   // no data at all
   } else {
      npy_intp dims[1] = {0};
      return (PyArrayObject*) PyArray_ZEROS(1, dims, PyArray_FLOAT, 0);
   }
}

static PyObject* boostnano_decode(PyObject *self, PyObject *args)
{
    PyArrayObject *array;
    PyObject * return_tuple;
    return_tuple = PyTuple_New(3);
    if (!PyArg_ParseTuple(args, "O", &array))
        return NULL;

    if (array->nd != 2 || ((array->descr->type_num != PyArray_DOUBLE) && (array->descr->type_num != PyArray_FLOAT32))) 
    {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type float");
        return NULL;
    }
    int signal_n=array->dimensions[1];
    int class_n=array->dimensions[0];
    vector<vector<double> > decoded(class_n,vector<double>(signal_n,0.0));
    vector<vector<short> > path(class_n,vector<short>(signal_n,0));
    vector<long> locs;
    vaterbi_decoding(array, decoded,path,locs);
    PyTuple_SET_ITEM(return_tuple,0,(PyObject*)vector2D_to_nparray<double>(decoded,PyArray_DOUBLE));
    PyTuple_SET_ITEM(return_tuple,1,(PyObject*)vector2D_to_nparray<short>(path,PyArray_SHORT));
    PyTuple_SET_ITEM(return_tuple,2,(PyObject*)vector1D_to_nparray<long>(locs,PyArray_LONG));
    return return_tuple;
}

void vaterbi_decoding(PyArrayObject *array, vector< vector<double> >& decoded, vector< vector<short> >& path, vector<long>& locs)
{
    double temp;
    int signal_n = array->dimensions[1];
    int class_n = array->dimensions[0];
    double epsilon=1e-9; //A small value to prevent log(0)
    for (int i=0; i<class_n; i++)
    {
        for (int j=0; j<signal_n; j++)
        {
            if(j==0||i==0)
            {
                path[i][j]=0;
                decoded[i][j] = decoded[i][max(j-1,0)] + log(*(double *)(array->data + i * array->strides[0] + j * array->strides[1])+epsilon);
                //Initialize the t=0 and class 0 probability
            }
            else
            {
                decoded[i][j] = decoded[i][j-1] + log(*(double *)(array->data + i * array->strides[0] + j * array->strides[1])+epsilon);
                temp = decoded[i-1][j-1] + log(*(double *)(array->data + i * array->strides[0] + j * array->strides[1])+epsilon);
                //The state is only allowed to transfer to the same class or next class.
                if (temp > decoded[i][j])
                {
                    decoded[i][j] = temp;
                    path[i][j] = i-1;
                }
                else
                {
                    path[i][j] = i;
                }
            }
        }
    }
    int last_idx = 0;
    temp = decoded[last_idx][signal_n-1];
    for (int i=1; i<class_n ;i++)
    {
        if ((decoded[i][signal_n-1]>temp) && (decoded[i][signal_n-1]!=0))
        {
            temp=decoded[i][signal_n-1];
            last_idx=i;
        }
    }
    
    for (int i=signal_n-1; i>=0; i--)
    {
        if (path[last_idx][i] != last_idx)
            locs.push_back((long)i);
        last_idx = path[last_idx][i];
    }//backtraking to get the segmentation locations
}

static char vaterbi_decoding_docs[] = "Decode: Applying a Vaterbi decoding of the input probabiltiy matrix.\n";

static PyMethodDef DecodeMethod[] = {
    {"decode", 
    boostnano_decode,
    METH_VARARGS,
    vaterbi_decoding_docs},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef boostnanomodule = {
	PyModuleDef_HEAD_INIT,
	"hmm", //the module name
	vaterbi_decoding_docs,
	-1,
	DecodeMethod
};

PyMODINIT_FUNC
PyInit_hmm(void){
    import_array();
    return PyModule_Create(&boostnanomodule);
};
