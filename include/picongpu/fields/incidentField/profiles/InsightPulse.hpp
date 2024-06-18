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
#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/memory/buffers/HostBuffer.hpp"

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

                        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                        pmacc::DataSpace<simDim> const globalSizePIC = subGrid.getGlobalDomain().size;  // global extent
                    };

                    /** InsightPulse incident E functor
                     *
                     * @tparam T_Params parameters
                     */
                    template<typename T_Params>
                    struct InsightPulseFunctorIncidentE
                        : public InsightPulseUnitless<T_Params>
                        // Todo: BaseFunctor einbinden, damit Funktionen wie Zeit, Ort etc. abfragen möglich ist
                    {
                        //! Unitless parameters type
                        using Unitless = InsightPulseUnitless<T_Params>;

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE InsightPulseFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
                            //: Base(currentStep, unitField)
                        {
                            // checkUnit(unitField);
                            loadFile();  // Gibt es eine bessere Stelle?
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


                        //! Load E and B field values from OpenPMD file to device buffer
                        void loadFile()
                        {
                            auto& dc = Environment<>::get().DataConnector();
                            /* Open a series (this does not read the dataset itself).
                             * This is MPI collective and so has to be done by all ranks.
                             */
                            auto& gc = Environment<simDim>::get().GridController();
                            auto const filename = getFilename();
                            log<picLog::PHYSICS>("Loading field values from file \"%1%\"")
                                % filename;

                            auto series
                                = ::openPMD::Series{filename, ::openPMD::Access::READ_ONLY, gc.getCommunicator().getMPIComm()};
                            auto meshE = series.iterations[Unitless::iteration].meshes[Unitless::datasetEName];
                            auto meshB = series.iterations[Unitless::iteration].meshes[Unitless::datasetBName];

                            // dataOrder und Achsennamen abfragen + sanity check wie in IndexConverter
                            ::openPMD::MeshRecordComponent datasetEx = meshE["x"];  // Polarisation direction E
                            ::openPMD::MeshRecordComponent datasetBy = meshB["y"];  // Polarisation direction main component B
                            ::openPMD::MeshRecordComponent datasetBz = meshB["z"];  // Small B component & Propagation direction

                            auto const indexConverterE = IndexConverter{meshE};
                            auto const indexConverterB = IndexConverter{meshB};

                            // Dataset extent
                            ::openPMD::Extent const datasetEExtent = datasetEx.getExtent();
                            ::openPMD::Extent const datasetBExtent = datasetBy.getExtent();

                            // grid spacing
                            auto const cellSizeOpenPMD = meshE.second.gridSpacing<float>();

                            DataSpace<simDim> EExtent, BExtent;
                            for(uint32_t d = 0u; d < simDim; d++)
                            {
                                // type conversion: openPMD::Extent to DataSpace<simDim>
                                EExtent[d] = datasetEExtent[d];
                                BExtent[d] = datasetBExtent[d];
                            }

                            using DataType = float_64; //datasetEx.getDatatype();
                            // error: datasetEx is ot a typename

                            // Field Buffer for Ex, By, Bz
                            auto ExfieldBuffer = pmacc::HostDeviceBuffer<DataType, simDim>(EExtent);
                            auto ByfieldBuffer = pmacc::HostDeviceBuffer<DataType, simDim>(BExtent);
                            auto BzfieldBuffer = pmacc::HostDeviceBuffer<DataType, simDim>(BExtent);

                            auto dataEx = std::shared_ptr<DataType>{nullptr};
                            auto dataBy = std::shared_ptr<DataType>{nullptr};
                            auto dataBz = std::shared_ptr<DataType>{nullptr};

                            bool readFromFile = true;  // can be used for sanity checks, otherwise dump it
                            if(readFromFile)
                            {
                                // load the whole data chunk, no slicing necessary
                                dataEx = datasetEx.loadChunk<DataType>();
                                dataBy = datasetBy.loadChunk<DataType>();
                                dataBz = datasetBz.loadChunk<DataType>();
                            }
                            // This is MPI collective and so has to be done by all ranks
                            series.flush();

                            if(readFromFile)  // filling host data boxes
                            {
                                auto const numEElements = std::accumulate(
                                std::begin(datasetEExtent),
                                std::end(datasetEExtent),
                                1u,
                                std::multiplies<uint32_t>());

                                auto hostExDataBox = ExfieldBuffer.getHostBuffer().getDataBox();

                                for(uint32_t linearIdx = 0u; linearIdx < numEElements; linearIdx++)
                                {
                                    // .get() seems to return a flat vector, has to be reshaped to fill dataBoxes
                                    auto const idx = indexConverterE.linearToXyz(linearIdx, datasetEExtent);
                                    hostExDataBox(idx) = dataEx.get()[linearIdx] * datasetEx.unitSI() / UNIT_EFIELD;
                                }

                                auto const numBElements = std::accumulate(
                                std::begin(datasetBExtent),
                                std::end(datasetBExtent),
                                1u,
                                std::multiplies<uint32_t>());

                                auto hostByDataBox = ByfieldBuffer.getHostBuffer().getDataBox();
                                auto hostBzDataBox = BzfieldBuffer.getHostBuffer().getDataBox();

                                for(uint32_t linearIdx = 0u; linearIdx < numBElements; linearIdx++)
                                {
                                    auto const idx = indexConverterB.linearToXyz(linearIdx, datasetBExtent);
                                    hostByDataBox(idx) = dataBy.get()[linearIdx] * datasetBy.unitSI() / UNIT_BFIELD;
                                    hostBzDataBox(idx) = dataBz.get()[linearIdx] * datasetBz.unitSI() / UNIT_BFIELD;
                                }
                            }

                            // series.close();  // Notwendig? Error: has no member close

                            // Copy host data to the device
                            ExfieldBuffer.hostToDevice();
                            eventSystem::getTransactionEvent().waitForFinished();
                            ByfieldBuffer.hostToDevice();
                            eventSystem::getTransactionEvent().waitForFinished();
                            BzfieldBuffer.hostToDevice();
                            eventSystem::getTransactionEvent().waitForFinished();
                            log<picLog::PHYSICS>("HostToDevice sucessful");
                        }  // loadFile()

                        //! Get file name to load density from
                        std::string getFilename() const
                        {
                            return Unitless::filename;
                        }

                        class IndexConverter
                        // ist so aus FromOpenPMDImpl.hpp übernommen, vllt geht ja auch der Zugriff über picongpu::densityProfiles::detail::IndexConverter
                        // Vielleicht ist es sinnvoller, das hier zu behalten und auf meinen Fall zu spezifizieren /vereinfachen
                        {
                        public:
                            /** Create an index converter for the given mesh
                            *
                            * @param mesh openPMD API mesh
                            */
                            IndexConverter(::openPMD::Mesh const& mesh)
                            {
                                if(mesh.dataOrder() != ::openPMD::Mesh::DataOrder::C)
                                    throw std::runtime_error(
                                        "Unsupported dataOrder in FromOpenPMD density dataset, only C is supported");
                                auto axisLabels = std::vector<std::string>{mesh.axisLabels()};
                                // When the attribute is not set, openPMD API currently makes it a vector of single "x"
                                if(axisLabels.size() <= 1)
                                    axisLabels = std::vector<std::string>{"x", "y", "z"};
                                std::array<std::string, 3> supportedAxes = {"x", "y", "z"};
                                for(auto d = 0; d < simDim; d++)
                                {
                                    auto it = std::find(begin(supportedAxes), begin(supportedAxes) + simDim, axisLabels[d]);
                                    if(it != std::end(supportedAxes))
                                    {
                                        openPMDAxisIndex[d] = std::distance(begin(supportedAxes), it);
                                        xyzAxisIndex[openPMDAxisIndex[d]] = d;
                                    }
                                    else
                                        throw std::runtime_error(
                                            "Unsupported axis label " + axisLabels[d] + " in FromOpenPMD density dataset");
                                }
                            }

                            /** Convert a multidimentional index from x-y-z to the openPMD coordinates
                            *
                            * @tparam T_Vector vector type, compatible to std::vector
                            *
                            * @param vector input vector
                            */
                            template<typename T_Vector>
                            T_Vector xyzToOpenPMD(T_Vector const& vector) const
                            {
                                auto result = vector;
                                for(auto d = 0; d < simDim; d++)
                                    result[openPMDAxisIndex[d]] = vector[d];
                                return result;
                            }

                            /** Convert a multidimentional index from openPMD to the x-y-z coordinates
                            *
                            * @tparam T_Vector vector type, compatible to std::vector
                            *
                            * @param vector input vector
                            */
                            template<typename T_Vector>
                            T_Vector openPMDToXyz(T_Vector const& vector) const
                            {
                                auto result = vector;
                                for(int32_t d = 0; d < simDim; d++)
                                    result[xyzAxisIndex[d]] = vector[d];
                                return result;
                            }

                            /** Convert a linear index in openPMD chunk to a multidimentional x-y-z index.
                            *
                            * @param openPMDLinearIndex linear index inside openPMD chunk
                            * @param xyzChunkExtent multidimentional chunk extent in xyz
                            */
                            pmacc::DataSpace<simDim> linearToXyz(uint32_t openPMDLinearIndex, ::openPMD::Extent xyzChunkExtent)
                                const
                            {
                                // Convert xyz extent to openPMD one
                                //auto const openPMDChunkExtent = xyzToOpenPMD(xyzChunkExtent);
                                // This is index in the openPMD coordinate system, the calculation relies on the C data order
                                pmacc::DataSpace<simDim> openPMDIdx;
                                auto tmpIndex = openPMDLinearIndex;
                                for(int32_t d = simDim - 1; d >= 0; d--)
                                {
                                    openPMDIdx[d] = tmpIndex % xyzChunkExtent[d];
                                    tmpIndex /= xyzChunkExtent[d];
                                }
                                // Now we convert it to the xyz coordinates
                                return openPMDIdx; //openPMDToXyz(openPMDIdx);
                            }

                        private:
                            // openPMDAxisIndex[0] is openPMD axis index for x, [1] - for y, [2] - for z
                            pmacc::DataSpace<simDim> openPMDAxisIndex;

                            // xyzAxisIndex[0] is x axis index in openPMD, [1] - y, [2] - z
                            pmacc::DataSpace<simDim> xyzAxisIndex;
                        };  // IndexConverter

                        /** Linear interpolation of scalar field data stored on a regular grid to PIConGPU-intern grid
                         * Extrapolation to default values
                         * Kann man im Idealfall für B wiederverwenden
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         */
                        HDINLINE float_X GridInterpolator(floatD_X const& totalCellIdx) const  //
                        {
                            auto const posPIC = totalCellIdx * cellSize;   // point at which to evaluate field data
                            auto const timePIC = this->currentStep * DELTA_T;
                            // warning: calling a __host__ function from a __host__ __device__ function is not allowed
                            //const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                            //auto globalSizePIC = subGrid.getGlobalDomain().size;  // global extent
                            auto test = datasetEExtent[0]*2.0;

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