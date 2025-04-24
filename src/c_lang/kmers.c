#include "Python.h"

static PyObject * get_kmers(PyObject *self, PyObject *args) {
    const char *sequence;
    int k;

    if (!PyArg_ParseTuple(args, "si", &sequence, &k)) {
        return NULL;
    }

    int sequence_length = strlen(sequence);
    int num_kmers = sequence_length - k + 1;

    PyObject *kmers_list = PyList_New(num_kmers);
    if (kmers_list == NULL) {
        return NULL;
    }

    char *kmer = (char *) malloc((k + 1) * sizeof(char));
    if (kmer == NULL) {
        Py_DECREF(kmers_list);
        return PyErr_NoMemory(); 
    }
    for (int i = 0; i < num_kmers; i++) {
        memcpy(kmer, sequence + i, k);
        kmer[k] = '\0';

        PyObject *py_kmer = PyUnicode_FromString(kmer);
        if (py_kmer == NULL) {
            free(kmer);
            Py_DECREF(kmers_list); 
            return NULL;
        }
        
        PyList_SetItem(kmers_list, i, py_kmer);
    }
    free(kmer);
    return kmers_list;
}

static PyMethodDef module_functions[] = {
    {"get_kmers", get_kmers, METH_VARARGS, "Generate k-mers from a given sequence."},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "kmers",  
    "Kmer generation Module", 
    -1,             // Stateless - maybe change this for threads
    module_functions
};

PyMODINIT_FUNC PyInit_kmers(void) {
    return PyModule_Create(&module);
}