# distutils: language=c++
import cython
from libcpp.vector cimport vector
import sys
from cpython cimport PyObject, PyList_New, PyLong_FromVoidPtr, PyList_SET_ITEM, Py_INCREF
# from cpython cimport PyListObject
# from cpython.list cimport PyListObject, PyList_New, PyList_GET_ITEM, Py_INCREF, PyObject, PyLong_FromVoidPtr
from libcpp.vector cimport vector
from cpython.ref cimport PyObject, Py_INCREF
from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.long cimport PyLong_FromVoidPtr

cdef class MinMaxStatsList:
    @cython.binding
    def __cinit__(self, int num):
        self.cmin_max_stats_lst = new CMinMaxStatsList(num)

    @cython.binding
    def set_delta(self, float value_delta_max):
        self.cmin_max_stats_lst[0].set_delta(value_delta_max)

    def __dealloc__(self):
        del self.cmin_max_stats_lst

cdef class ResultsWrapper:

    @cython.binding
    def __cinit__(self, int num):
        self.cresults = CSearchResults(num)
    
    @cython.binding
    def get_search_len(self):
        return self.cresults.search_lens

    @cython.binding
    def get_nodes(self):
        cdef vector[CNode*] nodes = self.cresults.nodes
        cdef Py_ssize_t i
        # cdef PyObject *node_list = PyList_New(nodes.size())
        cdef PyObject *node_list = <PyObject *>PyList_New(<Py_ssize_t>nodes.size())
        
        if node_list is NULL:
            return NULL
        for i in range(nodes.size()):
            PyList_SET_ITEM(node_list, i, PyLong_FromVoidPtr(<void*>nodes[i]))
        # Increase reference count of the object before returning
        Py_INCREF(node_list)
        return <object>node_list
        # return node_list
        # return <object>PyListObject.from_pyobject(node_list)
        # return <object>node_list
        # return <PyObject *>node_list

    # TODO(pu)
    # @cython.binding
    # def get_results_nodes(self):
    #    return self.cresults.nodes

    # @cython.binding
    # def get_results_search_paths(self):
    #    return self.cresults.search_paths

    # @cython.binding
    # def get_results(self):
    #    return self.cresults
    

cdef class Roots:
    @cython.binding
    def __cinit__(self, int root_num, vector[vector[int]] legal_actions_list):
        self.root_num = root_num
        self.roots = new CRoots(root_num, legal_actions_list)

    @cython.binding
    def prepare(self, float root_exploration_fraction, list noises, list value_prefix_pool,
                list policy_logits_pool, vector[int] & to_play_batch):
        self.roots[0].prepare(root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool, to_play_batch)

    @cython.binding
    def prepare_no_noise(self, list value_prefix_pool, list policy_logits_pool, vector[int] & to_play_batch):
        self.roots[0].prepare_no_noise(value_prefix_pool, policy_logits_pool, to_play_batch)

    @cython.binding
    def get_trajectories(self):
        return self.roots[0].get_trajectories()

    @cython.binding
    def get_distributions(self):
        return self.roots[0].get_distributions()

    @cython.binding
    def get_values(self):
        return self.roots[0].get_values()

    # visualize related code
    #def get_root(self, int index):
    #    return self.roots[index]

    @cython.binding
    def clear(self):
        self.roots[0].clear()

    def __dealloc__(self):
        del self.roots

    @property
    def num(self):
        return self.root_num

cdef class Node:
    def __cinit__(self):
        pass

    def __cinit__(self, float prior, vector[int] & legal_actions):
        pass

    @cython.binding
    def expand(self, int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix,
               list policy_logits):
        cdef vector[float] cpolicy = policy_logits
        self.cnode.expand(to_play, hidden_state_index_x, hidden_state_index_y, value_prefix, cpolicy)

@cython.binding
def batch_backpropagate(int hidden_state_index_x, float discount_factor, list value_prefixs, list values, list policies,
                         MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list is_reset_lst,
                         list to_play_batch):
    cdef int i
    cdef vector[float] cvalue_prefixs = value_prefixs
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicies = policies

    cbatch_backpropagate(hidden_state_index_x, discount_factor, cvalue_prefixs, cvalues, cpolicies,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, is_reset_lst, to_play_batch)

@cython.binding
def batch_traverse(Roots roots, int pb_c_base, float pb_c_init, float discount_factor, MinMaxStatsList min_max_stats_lst,
                   ResultsWrapper results, list virtual_to_play_batch):
    cbatch_traverse(roots.roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst.cmin_max_stats_lst,
                    results.cresults, virtual_to_play_batch)

    return results.cresults.hidden_state_index_x_lst, results.cresults.hidden_state_index_y_lst, results.cresults.last_actions, results.cresults.virtual_to_play_batchs


@cython.binding
def batch_traverse(Roots roots, int pb_c_base, float pb_c_init, float discount_factor, MinMaxStatsList min_max_stats_lst,
                   ResultsWrapper results, list virtual_to_play_batch):
    cbatch_traverse(roots.roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst.cmin_max_stats_lst,
                    results.cresults, virtual_to_play_batch)

    return results.cresults.hidden_state_index_x_lst, results.cresults.hidden_state_index_y_lst, results.cresults.last_actions, results.cresults.virtual_to_play_batchs


# @cython.binding
# def get_results(ResultsWrapper results):
#    return results.cresults

# @cython.binding
# def get_results_nodes(ResultsWrapper results):
#     return results.cresults.nodes

# @cython.binding
# def get_results_search_paths(ResultsWrapper results):
#    return results.cresults.search_path