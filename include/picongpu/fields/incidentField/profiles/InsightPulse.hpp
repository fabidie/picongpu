/* Copyright 2022 Fabia Dietrich
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/incidentField/Functors.hpp"
#include "picongpu/fields/incidentField/Traits.hpp"
#include "picongpu/fields/incidentField/profiles/InsightPulse.def"
#include "pmacc/memory/buffers/HostDeviceBuffer.hpp"

#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/math/Complex.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <openPMD/openPMD.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace profiles
            {
                namespace detail
                {
                    /** Unitless InsightPulse parameters
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct InsightPulseUnitless : public T_Params
                    {
                        //! User SI parameters
                        using Params = T_Params;

                        static constexpr float_X defaultEFieldValue = static_cast<float_X>(Params::defaultEFieldValueSI / UNIT_EFIELD);
                        static constexpr float_X defaultBFieldValue = static_cast<float_X>(Params::defaultBFieldValueSI / UNIT_BFIELD);

                        // Propagation direction, works just in 3D parallel to one axis
                        static constexpr float_X DIR_X = static_cast<float_X>(Params::DIRECTION_X);
                        static constexpr float_X DIR_Y = static_cast<float_X>(Params::DIRECTION_Y);
                        static constexpr float_X DIR_Z = static_cast<float_X>(Params::DIRECTION_Z);

                        // Polarization direction vector -> cyclic swap
                        static constexpr float_X POL_DIR_X = static_cast<float_X>(Params::DIRECTION_Z);
                        static constexpr float_X POL_DIR_Y = static_cast<float_X>(Params::DIRECTION_X);
                        static constexpr float_X POL_DIR_Z = static_cast<float_X>(Params::DIRECTION_Y);

                        /** Max amplitude of E field
                         *
                         * unit: UNIT_EFIELD
                         * Braucht man für GetAmplitude in Traits.hpp
                         * Später: dort für <InsightPulse> Profil spezifizieren
                         */
                        static constexpr float_X AMPLITUDE = 1.0_X;

                        /** Wave length along propagation direction
                         *
                         * unit: UNIT_LENGTH
                         * Braucht man für GetPhaseVelocity in Traits.hpp
                         * Später: dort für <InsightPulse> Profil spezifizieren
                         */
                        static constexpr float_X WAVE_LENGTH
                            = static_cast<float_X>(800.0e-9 / UNIT_LENGTH);
                    };

                    template<typename T_Params>
                    struct OpenPMDdata
                       : public InsightPulseUnitless<T_Params>
                    {
                         //! Unitless parameters type
                        using Unitless = InsightPulseUnitless<T_Params>;

                        //! HostDeviceBuffer to store OpenPMD data in
                        std::shared_ptr<pmacc::HostDeviceBuffer<float_64, simDim>> fieldBufferEx;

                        static OpenPMDdata& get()
                        {
                            static OpenPMDdata dataBuffer{};
                            return dataBuffer;
                        }
                    private:
                        OpenPMDdata()  // = loading the file and pushing it to device
                        {
                            // allocate buffer

                            auto& dc = Environment<>::get().DataConnector();  // wozu brauche ich den?
                            /* Open a series (this does not read the dataset itself).
                             * This is MPI collective and so has to be done by all ranks.
                             */
                            auto& gc = Environment<simDim>::get().GridController();

                            auto series
                                = ::openPMD::Series{Unitless::filename, ::openPMD::Access::READ_ONLY, gc.getCommunicator().getMPIComm()};
                            auto meshE = series.iterations[Unitless::iteration].meshes[Unitless::datasetEName];

                            // dataOrder und Achsennamen abfragen + sanity check wie in IndexConverter
                            ::openPMD::MeshRecordComponent datasetEx = meshE["x"];  // Polarisation direction E

                            // Dataset extent
                            ::openPMD::Extent const datasetEExtent = datasetEx.getExtent();

                            // grid spacing
                            //auto const cellSizeOpenPMD = meshE.gridSpacing<>();

                            DataSpace<simDim> EExtent;
                            for(uint32_t d = 0u; d < simDim; d++)
                                EExtent[d] = datasetEExtent[d];  // type conversion: openPMD::Extent to DataSpace<simDim>

                            // besser: DataType abfragen!
                            using DataType = float_64; //datasetEx.getDatatype();
                            // error: datasetEx is not a typename

                            fieldBufferEx = std::make_shared<pmacc::HostDeviceBuffer<DataType, simDim>>(EExtent);  // oben instanziiert
                            //dc.share(fieldBufferEx);  // brauche ich das hier?
                            // error: no suitable user-defined conversion

                            auto dataEx = std::shared_ptr<DataType>{nullptr};

                            bool readFromFile = true;  // can be used for sanity checks, otherwise dump it
                            if(readFromFile)
                            {
                                // load the whole data chunk, no slicing necessary
                                dataEx = datasetEx.loadChunk<DataType>();
                            }
                            // This is MPI collective and so has to be done by all ranks
                            series.flush();

                            if(readFromFile)  // filling host data boxes
                            {
                                // test print
                                for( size_t col = 0;
                                    col < datasetEExtent[1] && col < 5;
                                    ++col )
                                    log<picLog::PHYSICS>("%1%, ") % dataEx.get()[col];

                                auto const numEElements = std::accumulate(
                                std::begin(datasetEExtent),
                                std::end(datasetEExtent),
                                1u,
                                std::multiplies<uint32_t>());

                                auto hostExDataBox = fieldBufferEx->getHostBuffer().getDataBox();

                                for(uint32_t linearIdx = 0u; linearIdx < numEElements; linearIdx++)
                                {
                                    // .get() seems to return a flat vector, has to be reshaped to fill dataBoxes
                                    pmacc::DataSpace<simDim> openPMDIdx;
                                    auto tmpIndex = linearIdx;
                                    for(int32_t d = simDim - 1; d >= 0; d--)
                                    {
                                        openPMDIdx[d] = tmpIndex % datasetEExtent[d];
                                        tmpIndex /= datasetEExtent[d];
                                    }
                                    hostExDataBox(openPMDIdx) = dataEx.get()[linearIdx] * datasetEx.unitSI() / UNIT_EFIELD;
                                }
                            }

                            // series.close();  // Notwendig? Error: has no member close

                            // Copy host data to the device
                            fieldBufferEx->hostToDevice();
                            eventSystem::getTransactionEvent().waitForFinished();
                            log<picLog::PHYSICS>("HostToDevice successful");
                        }
                    };

                    /** InsightPulse incident E functor
                     *
                     * @tparam T_Params parameters
                     */
                    template<typename T_Params>
                    struct InsightPulseFunctorIncidentE
                        : public InsightPulseUnitless<T_Params>
                    {
                        //! Unitless parameters type
                        using Unitless = InsightPulseUnitless<T_Params>;

                        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                        pmacc::DataSpace<simDim> const globalSizePIC = subGrid.getGlobalDomain().size;  // global extent

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE InsightPulseFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
                            : timePIC(currentStep * DELTA_T)
                        {
                            // checkUnit(unitField);
                            auto dataBuffer& = OpenPMDdata<T_Params>::get();
                            devDataBox = dataBuffer.getDeviceDataBox();  // die muss ich doch auch irgendwo vorher schon instanziiert haben, damit ich außerhalb des Konstruktors Zugriff habe, oder?
                        }

                        /** Read incident field E value for the given position and time step
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @return incident field E value in internal units
                         */
                        HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
                        {
                            return getPolarisationVector() * getValueE(totalCellIdx);
                        }

                        //! Polarisation vector
                        HDINLINE static constexpr float3_X getPolarisationVector()
                        {
                            return float3_X(Unitless::POL_DIR_X, Unitless::POL_DIR_Y, Unitless::POL_DIR_Z);
                        }

                        /** Linear interpolation of scalar field data stored on a regular grid to PIConGPU-intern grid
                         * Extrapolation to default values
                         * Kann man im Idealfall für B wiederverwenden
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         */
                        HDINLINE float_X GridInterpolator(floatD_X const& totalCellIdx) const
                        {
                            auto const posPIC = totalCellIdx * cellSize;   // point at which to evaluate field data
                            // auto const timePIC = this->currentStep * DELTA_T;
                            //  error: class "picongpu::fields::incidentField::profiles::detail::InsightPulseFunctorIncidentE<picongpu::fields::incidentField::profiles::defaults::InsightPulseParam>"
                            //has no member "currentStep"
                            // warning: calling a __host__ function from a __host__ __device__ function is not allowed
                            //const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                            //auto globalSizePIC = subGrid.getGlobalDomain().size;  // global extent
                            auto test = timePIC * 2.0;

                            return 0.0;
                        }

                    private:
                        /** Get value of E field for the given position, using linear interpolation of OpenPMD-stored field data
                         * Extrapolation to default values
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         */
                        HDINLINE float_X getValueE(floatD_X const& totalCellIdx) const
                        {
                            float_X E = GridInterpolator(totalCellIdx);
                            return E;
                        } // getValueE

                    protected:  // damit B Functor auch Zugriff hat
                        /** Current time for calculating the field at the origin
                         */
                        float_X const timePIC;
                        pmacc::Buffer<float_64, simDim>::DataBoxType devDataBox;

                    }; // DispersivePulseFunctorIncidentE

                    /** InsightPulse incident B functor
                     *
                     *
                     * @tparam T_Params parameters
                     */
                    template<typename T_Params>
                    struct InsightPulseFunctorIncidentB : public InsightPulseUnitless<T_Params>
                    {
                        //! Unitless parameters type
                        using Unitless = InsightPulseUnitless<T_Params>;

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldB_internal = fieldB_SI / unitField
                         */
                        HINLINE InsightPulseFunctorIncidentB(float_X const currentStep, float3_64 const unitField)
                        {
                        }

                        /** Calculate incident field B value for the given position
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @return incident field B value in internal units
                         */
                        HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
                        {
                            return float3_X(0.0_X, 0.0_X, 0.0_X);
                        }
                    }; // InsightPulseFunctorIncidentB
                } // namespace detail

                template<typename T_Params>
                struct InsightPulse
                {
                    //! Get text name of the incident field profile
                    static HINLINE std::string getName()
                    {
                        return "InsightPulse";
                    }
                };
            } // namespace profiles

            namespace detail
            {
                /** Get type of incident field E functor for the Insight laser profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentE<profiles::InsightPulse<T_Params>>
                {
                    using type = profiles::detail::InsightPulseFunctorIncidentE<T_Params>;
                };

                /** Get type of incident field B functor for the dispersive laser profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentB<profiles::InsightPulse<T_Params>>
                {
                    using type = profiles::detail::InsightPulseFunctorIncidentB<T_Params>;
                };

            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu