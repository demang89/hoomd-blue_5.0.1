// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __POTENTIAL_PAIR_DLVOStokesDrag_H__
#define __POTENTIAL_PAIR_DLVOStokesDrag_H__

#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>
#include <stdexcept>

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include "NeighborList.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/Variant.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#include "hoomd/ManagedArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/managed_allocator.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {

class PYBIND11_EXPORT PotentialPairDLVOStokesDrag : public ForceCompute
    {
    public:
    struct param_type
        {
        Scalar kappa;
        Scalar Z;
        Scalar A;
        Scalar a1;
        Scalar a2;
        Scalar kn;

#ifdef ENABLE_HIP
        void set_memory_hint() const {}
#endif

        void load_shared(char*& ptr, unsigned int& available_bytes) { }

        void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

        param_type() : kappa(0), Z(0), A(0), kn(0) { }

#ifndef __HIPCC__
        param_type(pybind11::dict v, bool managed)
            {
            kappa = v["kappa"].cast<Scalar>();
            Z = v["Z"].cast<Scalar>();
            A = v["A"].cast<Scalar>();
            a1 = v["a1"].cast<Scalar>();
            a2 = v["a2"].cast<Scalar>();
            kn = v["kn"].cast<Scalar>();
            }

        pybind11::object toPython()
            {
            pybind11::dict v;
            v["kappa"] = kappa;
            v["Z"] = Z;
            v["A"] = A;
            v["a1"] = a1;
            v["a2"] = a2;
            v["kn"] = kn;
            return std::move(v);
            }
#endif
        }
#ifdef SINGLE_PRECISION
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    PotentialPairDLVOStokesDrag(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<NeighborList> nlist);
    virtual ~PotentialPairDLVOStokesDrag();

    virtual void setParams(unsigned int typ1, unsigned int typ2, const param_type& param);
    virtual void setParamsPython(pybind11::tuple typ, pybind11::object params);
    virtual pybind11::object getParamsPython(pybind11::tuple typ);

    virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);
    Scalar getRCut(pybind11::tuple types);
    virtual void setRCutPython(pybind11::tuple types, Scalar r_cut);

    virtual void setRon(unsigned int typ1, unsigned int typ2, Scalar ron);
    Scalar getROn(pybind11::tuple types);
    virtual void setROnPython(pybind11::tuple types, Scalar r_on);

    virtual void setGamma(Scalar gamma){m_gamma=gamma;}
    virtual Scalar getGamma(){return m_gamma;}

    virtual void setT(std::shared_ptr<Variant> T){m_T = T;}
    virtual std::shared_ptr<Variant> getT(){return m_T;}

    virtual void setRc(Scalar Rc){m_Rc=Rc;}
    virtual Scalar getRc(){return m_Rc;}

    virtual void setTypes(unsigned int types){m_types=types;}
    virtual unsigned int getTypes(){return m_types;}

    virtual void setCorrReqd(bool CorrReqd){m_CorrReqd=CorrReqd;}
    virtual bool getCorrReqd(){return m_CorrReqd;}

    virtual void validateTypes(unsigned int typ1, unsigned int typ2, std::string action);

    enum energyShiftMode
        {
        no_shift = 0,
        shift,
        };

    void setShiftMode(energyShiftMode mode)
        {
        m_shift_mode = mode;
        }

    void setShiftModePython(std::string mode)
        {
        if (mode == "none")
            {
            m_shift_mode = no_shift;
            }
        else if (mode == "shift")
            {
            m_shift_mode = shift;
            }
        else
            {
            throw std::runtime_error("Invalid energy shift mode.");
            }
        }

    std::string getShiftMode()
        {
        switch (m_shift_mode)
            {
        case no_shift:
            return "none";
        case shift:
            return "shift";
        default:
            return "";
            }
        }

    virtual void notifyDetach()
        {
        if (m_attached)
            {
            m_nlist->removeRCutMatrix(m_r_cut_nlist);
            }
        m_attached = false;
        }

#ifdef ENABLE_MPI
    virtual CommFlags getRequestedCommFlags(uint64_t timestep);
#endif

    protected:
    std::shared_ptr<NeighborList> m_nlist; //!< The neighborlist to use for the computation
    energyShiftMode m_shift_mode; //!< Store the mode with which to handle the energy shift at r_cut
    Index2D m_typpair_idx;        //!< Helper class for indexing per type pair arrays
    GPUArray<Scalar> m_rcutsq; //!< Cutoff radius squared per type pair
    GPUArray<Scalar> m_ronsq;
    std::vector<param_type, hoomd::detail::managed_allocator<param_type>>m_params; //!< Pair parameters per type pair
    bool m_attached = true;
    std::shared_ptr<GPUArray<Scalar>> m_r_cut_nlist;
    std::vector<unsigned int> m_num_particles_by_type;
    std::shared_ptr<Variant> m_T;
    Scalar m_gamma;
    Scalar m_Rc;
    unsigned int m_types;
    bool m_CorrReqd;
    virtual void computeForces(uint64_t timestep);
#ifdef ENABLE_MPI
    std::shared_ptr<Communicator> m_comm;
#endif
    };
    } // end namespace md
    } // end namespace hoomd

#endif // __POTENTIAL_PAIR_DLVOStokesDrag_H__
