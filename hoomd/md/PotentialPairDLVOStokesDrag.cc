// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.
#include "PotentialPairDLVOStokesDrag.h"
#include<iostream>

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
*/
PotentialPairDLVOStokesDrag::PotentialPairDLVOStokesDrag(std::shared_ptr<SystemDefinition> sysdef,
                                                        std::shared_ptr<NeighborList> nlist)
    : ForceCompute(sysdef), m_nlist(nlist), m_shift_mode(no_shift),
      m_typpair_idx(m_pdata->getNTypes())
    {
    m_exec_conf->msg->notice(5) << "Constructing PotentialPairDLVOStokesDrag" << std::endl;
    assert(m_pdata);
    assert(m_nlist);

    GPUArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), m_exec_conf);
    m_rcutsq.swap(rcutsq);
    GPUArray<Scalar> ronsq(m_typpair_idx.getNumElements(), m_exec_conf);
    m_ronsq.swap(ronsq);
    m_params = std::vector<param_type, hoomd::detail::managed_allocator<param_type>>(
        m_typpair_idx.getNumElements(),
        param_type(),
        hoomd::detail::managed_allocator<param_type>(m_exec_conf->isCUDAEnabled()));

    m_r_cut_nlist = std::make_shared<GPUArray<Scalar>>(m_typpair_idx.getNumElements(), m_exec_conf);
    nlist->addRCutMatrix(m_r_cut_nlist);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled())
        {
        cudaMemAdvise(m_params.data(),
                      m_params.size() * sizeof(param_type),
                      cudaMemAdviseSetReadMostly,
                      0);
        cudaMemPrefetchAsync(m_params.data(),
                             sizeof(param_type) * m_params.size(),
                             m_exec_conf->getGPUId());

        }
#endif

    // get number of each type of particle, needed for energy and pressure correction
    m_num_particles_by_type.resize(m_pdata->getNTypes());
    std::fill(m_num_particles_by_type.begin(), m_num_particles_by_type.end(), 0);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int typeid_i = __scalar_as_int(h_postype.data[i].w);
        m_num_particles_by_type[typeid_i] += 1;
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // reduce number of each type of particle on all processors
        MPI_Allreduce(MPI_IN_PLACE,
                      m_num_particles_by_type.data(),
                      m_pdata->getNTypes(),
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto comm_weak = m_sysdef->getCommunicator();
        assert(comm_weak.lock());
        m_comm = comm_weak.lock();
        }
#endif
    }

PotentialPairDLVOStokesDrag::~PotentialPairDLVOStokesDrag()
    {
    m_exec_conf->msg->notice(5) << "Destroying PotentialPairDLVOStokesDrag" << std::endl;

    if (m_attached)
        {
        m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param param Parameter to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   automatically set.
*/
void PotentialPairDLVOStokesDrag::setParams(unsigned int typ1,
                                                    unsigned int typ2,
                                                    const param_type& param)
    {
    validateTypes(typ1, typ2, "setting params");
    m_params[m_typpair_idx(typ1, typ2)] = param;
    m_params[m_typpair_idx(typ2, typ1)] = param;
    }

void PotentialPairDLVOStokesDrag::setParamsPython(pybind11::tuple typ,
                                                          pybind11::object params)
    {
    auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    setParams(typ1, typ2, param_type(params, m_exec_conf->isCUDAEnabled()));
    }

pybind11::object PotentialPairDLVOStokesDrag::getParamsPython(pybind11::tuple typ)
    {
    auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    validateTypes(typ1, typ2, "getting params");

    return m_params[m_typpair_idx(typ1, typ2)].toPython();
    }

void PotentialPairDLVOStokesDrag::validateTypes(unsigned int typ1,
                                                        unsigned int typ2,
                                                        std::string action)
    {
    // TODO change logic to just throw an exception
    auto n_types = this->m_pdata->getNTypes();
    if (typ1 >= n_types || typ2 >= n_types)
        {
        throw std::runtime_error("Error in" + action + " for pair potential. Invalid type");
        }
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param rcut Cutoff radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   automatically set.
*/
void PotentialPairDLVOStokesDrag::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    validateTypes(typ1, typ2, "setting r_cut");
        {
        // store r_cut**2 for use internally
        ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::readwrite);
        h_rcutsq.data[m_typpair_idx(typ1, typ2)] = rcut * rcut;
        h_rcutsq.data[m_typpair_idx(typ2, typ1)] = rcut * rcut;

        // store r_cut unmodified for so the neighbor list knows what particles to include
        ArrayHandle<Scalar> h_r_cut_nlist(*m_r_cut_nlist,
                                          access_location::host,
                                          access_mode::readwrite);
        h_r_cut_nlist.data[m_typpair_idx(typ1, typ2)] = rcut;
        h_r_cut_nlist.data[m_typpair_idx(typ2, typ1)] = rcut;
        }

    // notify the neighbor list that we have changed r_cut values
    m_nlist->notifyRCutMatrixChange();
    }

void PotentialPairDLVOStokesDrag::setRCutPython(pybind11::tuple types, Scalar r_cut)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    setRcut(typ1, typ2, r_cut);
    }

Scalar PotentialPairDLVOStokesDrag::getRCut(pybind11::tuple types)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateTypes(typ1, typ2, "getting r_cut.");
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    return sqrt(h_rcutsq.data[m_typpair_idx(typ1, typ2)]);
    }

void PotentialPairDLVOStokesDrag::setRon(unsigned int typ1, unsigned int typ2, Scalar ron)
    {
    validateTypes(typ1, typ2, "setting r_on");
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::readwrite);
    h_ronsq.data[m_typpair_idx(typ1, typ2)] = ron * ron;
    h_ronsq.data[m_typpair_idx(typ2, typ1)] = ron * ron;
    }

void PotentialPairDLVOStokesDrag::setROnPython(pybind11::tuple types, Scalar r_on)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    setRon(typ1, typ2, r_on);
    }

Scalar PotentialPairDLVOStokesDrag::getROn(pybind11::tuple types)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateTypes(typ1, typ2, "getting r_on");
    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::read);
    return sqrt(h_ronsq.data[m_typpair_idx(typ1, typ2)]);
    }

/*! \post The pair forces are computed for the given timestep. The neighborlist's compute method is
   called to ensure that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
void PotentialPairDLVOStokesDrag::computeForces(uint64_t timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::host,
                                        access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    //Index2D nli = m_nlist->getNListIndexer();
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
                                    access_location::host,
                                    access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::read);

    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                     access_location::host,
                                     access_mode::read);

    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);


    // force arrays
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    const BoxDim box = m_pdata->getGlobalBox();
    Scalar L_Y = box.getL().y;
    Scalar shear_rate = this->m_SR/L_Y;

    ArrayHandle<Scalar> h_ronsq(m_ronsq, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    memset(&h_force.data[0], 0, sizeof(Scalar4) * m_pdata->getN());
    memset(&h_virial.data[0], 0, sizeof(Scalar) * m_virial.getNumElements());

    uint16_t seed = this->m_sysdef->getSeed();
        // for each particle
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        // sanity check
        assert(typei < m_pdata->getNTypes());

        // initialize current particle force, torque, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar pei = Scalar(0.0);
        Scalar virialxxi = 0.0;
        Scalar virialxyi = 0.0;
        Scalar virialxzi = 0.0;
        Scalar virialyyi = 0.0;
        Scalar virialyzi = 0.0;
        Scalar virialzzi = 0.0;

        Scalar phi = Scalar(0.0);
        // loop over all of the neighbors of this particle
        const size_t myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());

            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < m_pdata->getNTypes());

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // get parameters for this type pair
            unsigned int typpair_idx = m_typpair_idx(typei, typej);
            const param_type& param = m_params[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];

            // design specifies that energies are shifted if
            // shift mode is set to shift
            bool energy_shift = false;
            if (m_shift_mode == shift)
                energy_shift = true;

            // compute the force and potential energy
            Scalar pair_eng = Scalar(0.0);
            Scalar force_divr = Scalar(0.0);

            bool evaluated=false;
            vec3<Scalar> rvec(dx);
            Scalar rsq = dot(rvec, rvec);
            Scalar rinv = fast::rsqrt(rsq);
            Scalar r = Scalar(1.0) / rinv;
            Scalar rcutinv = fast::rsqrt(rcutsq);
            Scalar rcut = Scalar(1.0) / rcutinv;

            if(r >= (m_Rc+param.a2))phi += Scalar(0.0);
            else if(r<(m_Rc-param.a2)) phi += Scalar(4.0) * pow(param.a2,3);
            else
                {
                Scalar h1 = (r*r+m_Rc*m_Rc-param.a2*param.a2)/(Scalar(2.0)*r);
                Scalar h2 = r-h1;
                Scalar vcap1 = pow((m_Rc-h1),2) * (h1+Scalar(2.0)*m_Rc);
                Scalar vcap2 = pow((param.a2-h2),2) * (h2+Scalar(2.0)*param.a2); 
                phi += (vcap1+vcap2);
                }

            bool executed = true;
            if(h_body.data[i]!= NO_BODY) executed = (h_body.data[i]!=h_body.data[j]);
            if (r < rcut && param.kappa != 0 && executed)
                {
                Scalar radsum = param.a1 + param.a2;
                Scalar radsub = param.a1 - param.a2;
                Scalar radprod = param.a1 * param.a2;
                Scalar radsumsq = param.a1 * param.a1 + param.a2 * param.a2;
                Scalar radsubsq = param.a1 * param.a1 - param.a2 * param.a2;

                Scalar rmin = Scalar(0.01) + radsum;
                if (r < radsum) force_divr = (radsum - r) * param.kn;
                if (r < rmin) r = rmin;
                Scalar rmds = r - radsum;
                Scalar rmdsqs = r * r - radsum * radsum;
                Scalar rmdsqm = r * r - radsub * radsub;
                Scalar radsuminv = Scalar(1.0) / radsum;
                Scalar rmdsqsinv = Scalar(1.0) / rmdsqs;
                Scalar rmdsqminv = Scalar(1.0) / rmdsqm;
                Scalar exp_val = fast::exp(-param.kappa * rmds);
                Scalar forcerep_divr = param.kappa * radprod * radsuminv * param.Z * exp_val / r;
                Scalar fatrterm1 = r * r * r * r + radsubsq * radsubsq - Scalar(2.0) * r * r * radsumsq;
                Scalar fatrterm1inv = Scalar(1.0) / fatrterm1 * Scalar(1.0) / fatrterm1;
                Scalar forceatr_divr
                    = -Scalar(32.0) * param.A / Scalar(3.0) * radprod * radprod * radprod * fatrterm1inv;
                force_divr += (forcerep_divr + forceatr_divr);

                Scalar engt1 = radprod * rmdsqsinv * param.A / Scalar(3.0);
                Scalar engt2 = radprod * rmdsqminv * param.A / Scalar(3.0);
                Scalar engt3 = slow::log(rmdsqs * rmdsqminv) * param.A / Scalar(6.0);
                pair_eng = r * forcerep_divr / param.kappa - engt1 - engt2 - engt3;
                if (energy_shift)
                    {
                    Scalar rcutt = rcut;
                    Scalar rmdscut = rcutt - radsum;
                    Scalar rmdsqscut = rcutt * rcutt - radsum * radsum;
                    Scalar rmdsqmcut = rcutt * rcutt - radsub * radsub;
                    Scalar rmdsqsinvcut = Scalar(1.0) / rmdsqscut;
                    Scalar rmdsqminvcut = Scalar(1.0) / rmdsqmcut;

                    Scalar engt1cut = radprod * rmdsqsinvcut * param.A / Scalar(3.0);
                    Scalar engt2cut = radprod * rmdsqminvcut * param.A / Scalar(3.0);
                    Scalar engt3cut = slow::log(rmdsqscut * rmdsqminvcut) * param.A / Scalar(6.0);
                    Scalar exp_valcut = fast::exp(-param.kappa * rmdscut);
                    Scalar forcerepcut_divr = param.kappa * radprod * radsuminv * param.Z * exp_valcut / rcutt;
                    pair_eng -= rcutt * forcerepcut_divr / param.kappa - engt1cut - engt2cut - engt3cut;
                    }
                evaluated=true;
                }


            if (evaluated)
                {
                Scalar force_div2r = force_divr * Scalar(0.5);

                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fi += dx * force_divr;
                pei += pair_eng * Scalar(0.5);
                if (compute_virial)
                    {
                    virialxxi += force_div2r * dx.x * dx.x;
                    virialxyi += force_div2r * dx.x * dx.y;
                    virialxzi += force_div2r * dx.x * dx.z;
                    virialyyi += force_div2r * dx.y * dx.y;
                    virialyzi += force_div2r * dx.y * dx.z;
                    virialzzi += force_div2r * dx.z * dx.z;
                    }

                // add the force to particle j if we are using the third law (MEM TRANSFER: 10
                // scalars / FLOPS: 8)
                if (third_law && j < m_pdata->getN())
                    {
                    unsigned int mem_idx = j;
                    h_force.data[mem_idx].x -= dx.x * force_divr;
                    h_force.data[mem_idx].y -= dx.y * force_divr;
                    h_force.data[mem_idx].z -= dx.z * force_divr;
                    h_force.data[mem_idx].w += pair_eng * Scalar(0.5);
                    if (compute_virial)
                        {
                        h_virial.data[0 * m_virial_pitch + mem_idx]
                            += force_div2r * dx.x * dx.x;
                        h_virial.data[1 * m_virial_pitch + mem_idx]
                            += force_div2r * dx.x * dx.y;
                        h_virial.data[2 * m_virial_pitch + mem_idx]
                            += force_div2r * dx.x * dx.z;
                        h_virial.data[3 * m_virial_pitch + mem_idx]
                            += force_div2r * dx.y * dx.y;
                        h_virial.data[4 * m_virial_pitch + mem_idx]
                            += force_div2r * dx.y * dx.z;
                        h_virial.data[5 * m_virial_pitch + mem_idx]
                            += force_div2r * dx.z * dx.z;
                        }
                    }
                }
            }
        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = i;
        unsigned int ptag = h_tag.data[mem_idx];
        if(typei==m_types)
            {
            phi = phi / (Scalar(4.0) * pow(m_Rc,3));
            Scalar cor_fac = Scalar(1.0);
            if(m_CorrReqd) cor_fac = pow((Scalar(1.0)-phi),2) / pow(Scalar(10.0),Scalar(1.82)*phi);
            hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::TwoStepLangevin, timestep, seed),
                                hoomd::Counter(ptag));
            hoomd::UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
            Scalar coeff = fast::sqrt((m_T->operator()(timestep) * m_gamma * cor_fac * Scalar(6.0))/this->m_deltaT);
            fi.x += (m_gamma*cor_fac*(shear_rate*h_pos.data[mem_idx].y-h_vel.data[mem_idx].x) + coeff*uniform(rng));
            fi.y += (-m_gamma*cor_fac*h_vel.data[mem_idx].y + coeff*uniform(rng));
            fi.z += (-m_gamma*cor_fac*h_vel.data[mem_idx].z + coeff*uniform(rng));
            }
        h_force.data[mem_idx].x += fi.x;
        h_force.data[mem_idx].y += fi.y;
        h_force.data[mem_idx].z += fi.z;
        h_force.data[mem_idx].w += pei;
        if (compute_virial)
            {
            h_virial.data[0 * m_virial_pitch + mem_idx] += virialxxi;
            h_virial.data[1 * m_virial_pitch + mem_idx] += virialxyi;
            h_virial.data[2 * m_virial_pitch + mem_idx] += virialxzi;
            h_virial.data[3 * m_virial_pitch + mem_idx] += virialyyi;
            h_virial.data[4 * m_virial_pitch + mem_idx] += virialyzi;
            h_virial.data[5 * m_virial_pitch + mem_idx] += virialzzi;
            }
        }
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
CommFlags PotentialPairDLVOStokesDrag::getRequestedCommFlags(uint64_t timestep)
    {
    CommFlags flags = CommFlags(0);
    flags[comm_flag::velocity] = 1;
    flags[comm_flag::tag] = 1;
    flags |= ForceCompute::getRequestedCommFlags(timestep);

    return flags;
    }
#endif

namespace detail
    {
//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/

void export_PotentialPairDLVOStokesDrag(pybind11::module& m)
    {
    pybind11::class_<PotentialPairDLVOStokesDrag, ForceCompute, std::shared_ptr<PotentialPairDLVOStokesDrag>>(m, "PotentialPairDLVOStokesDrag")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("setParams", &PotentialPairDLVOStokesDrag::setParamsPython)
        .def("getParams", &PotentialPairDLVOStokesDrag::getParamsPython)
        .def("setRCut", &PotentialPairDLVOStokesDrag::setRCutPython)
        .def("getRCut", &PotentialPairDLVOStokesDrag::getRCut)
        .def("setROn", &PotentialPairDLVOStokesDrag::setROnPython)
        .def("getROn", &PotentialPairDLVOStokesDrag::getROn)
        .def_property("gamma_d", &PotentialPairDLVOStokesDrag::getGamma, &PotentialPairDLVOStokesDrag::setGamma)
        .def_property("kT", &PotentialPairDLVOStokesDrag::getT, &PotentialPairDLVOStokesDrag::setT)
        .def_property("Rc", &PotentialPairDLVOStokesDrag::getRc, &PotentialPairDLVOStokesDrag::setRc)
        .def_property("types", &PotentialPairDLVOStokesDrag::getTypes, &PotentialPairDLVOStokesDrag::setTypes)
        .def_property("CorrReqd", &PotentialPairDLVOStokesDrag::getCorrReqd, &PotentialPairDLVOStokesDrag::setCorrReqd)
        .def_property("mode",
                      &PotentialPairDLVOStokesDrag::getShiftMode,
                      &PotentialPairDLVOStokesDrag::setShiftModePython);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
