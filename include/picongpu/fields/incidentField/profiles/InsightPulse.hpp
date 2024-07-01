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
#include <valarray>  // std::abs(valarray)
#include <memory>
#include <algorithm>

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

                        // Propagation direction, works just in 3D parallel to one axis
                        static constexpr float_X DIR_X = static_cast<float_X>(Params::DIRECTION_X);
                        static constexpr float_X DIR_Y = static_cast<float_X>(Params::DIRECTION_Y);
                        static constexpr float_X DIR_Z = static_cast<float_X>(Params::DIRECTION_Z);

                        // Polarization direction vector -> cyclic swap
                        // Anzahl zyklischer Vertauschungen aus OpenPMD faten auslesen
                        // NOCH NICHT KORREKT!
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
                    struct InsightPulseFunctorIncidentE;

                    /** Erklärung
                     *
                     */
                    template<typename T_Params>
                    struct OpenPMDdata
                       : public InsightPulseUnitless<T_Params>
                       //, public BaseFunctor?
                    {
                         //! Unitless parameters type
                        using Params = InsightPulseUnitless<T_Params>;

                        //! HostDeviceBuffer to store OpenPMD data in
                        // evtl reicht hostBuffer
                        std::shared_ptr<pmacc::HostDeviceBuffer<float_64, simDim>> fieldBufferE;

                        //! Necessary attributes
                        DataSpace<simDim> extentOpenPMD;  // dataSpacwe
                        float3_X cellSizeOpenPMD;
                        // warning #20094-D: a host member cannot be directly read in a device function
                        // std::shared_ptr<DataSpace<simDim>> extentOpenPMD;
                        // std::shared_ptr<float3_X> cellSizeOpenPMD;
                        // wie krieg ich die hier raus???
                        //! Storing Attributes in HostBuffer to permit access
                        // 1dimensional -> 1 statt simDim
                        // reicht HostBuffer?
                        std::shared_ptr<pmacc::HostDeviceBuffer<float_X, 1>> bufferExtentOpenPMD;  // ist eigentlich uint
                        std::shared_ptr<pmacc::HostDeviceBuffer<float_X, 1>> bufferCellSizeOpenPMD;

                        static OpenPMDdata& get()
                        {
                            static OpenPMDdata dataBuffer{};
                            return dataBuffer;
                        }
                    private:
                        OpenPMDdata()  // = loading the file
                        {
                            //! implement axis alignment!

                            /* Open a series (this does not read the dataset itself).
                             * This is MPI collective and so has to be done by all ranks.
                             */
                            auto& gc = Environment<simDim>::get().GridController();

                            auto series
                                = ::openPMD::Series{Params::filename, ::openPMD::Access::READ_ONLY, gc.getCommunicator().getMPIComm()};
                            ::openPMD::Mesh meshE = series.iterations[Params::iteration].meshes[Params::datasetEName];
                            // check data order
                            if(meshE.dataOrder() != ::openPMD::Mesh::DataOrder::C)
                                throw std::runtime_error(
                                    "Unsupported dataOrder in FromOpenPMD density dataset, only C is supported");
                            // get axis labels
                            auto const axisLabels = std::vector<std::string>{meshE.axisLabels()};  // ("x", "y", "z")
                            // evtl default axis labels if attribute is not set?
                            // DataBox alignment
                            float3_X const xyzAxisIndex{0, 1, 2};  // 2, 1, 0 wäre PIConGPu order, wäre das sinnvoller?
                            
                            // filling first with unknown axis index, then override the two known directions
                            float3_X aligningAxisIndex = float3_X::create(pmacc::math::abs(pmacc::math::dot(xyzAxisIndex,
                                        pmacc::math::cross(InsightPulseFunctorIncidentE<T_Params>::getDirection(), InsightPulseFunctorIncidentE<T_Params>::getPolarisationVector()))));
                                        // does the access to Efunctor work?  --> print test
                                        log<picLog::PHYSICS>("getDirection %1%") % InsightPulseFunctorIncidentE<T_Params>::getDirection();

                            // aligning propagation direction
                            // does std::find work even if axisLabels is not sorted?
                            auto it_prop = std::find(axisLabels.begin(), axisLabels.end(), Params::propagationAxisOpenPMD);
                            if(it_prop != std::end(axisLabels))
                                aligningAxisIndex[std::distance(begin(axisLabels), it_prop)]
                                    = pmacc::math::dot(xyzAxisIndex, InsightPulseFunctorIncidentE<T_Params>::getDirection());
                            //else
                                //throw std::runtime_error(
                                  //  "Could not find propagation axis %1% in OpenPMD dataset") % Params::propagationAxisOpenPMD;

                            // aligning polarisation direction
                            auto it_pola = std::find(axisLabels.begin(), axisLabels.end(), Params::polarisationAxisOpenPMD);
                            if(it_pola != std::end(axisLabels))
                                aligningAxisIndex[std::distance(begin(axisLabels), it_pola)]
                                    = pmacc::math::dot(xyzAxisIndex, InsightPulseFunctorIncidentE<T_Params>::getPolarisationVector());
                            //else
                              //  throw std::runtime_error(
                                //    "Could not find polarisation axis %1% in OpenPMD dataset") % Params::polarisationAxisOpenPMD;

                            log<picLog::PHYSICS>("aligningAxisIndex %1%") % aligningAxisIndex;

                            ::openPMD::MeshRecordComponent meshRecordE = meshE[Params::polarisationAxisOpenPMD];  // Polarisation direction record component
                            ::openPMD::Extent const extentRaw = meshRecordE.getExtent();
                            auto cellSizeRaw = meshE.gridSpacing<float_X>();

                            bufferExtentOpenPMD = std::make_shared<pmacc::HostDeviceBuffer<float_X, 1>>(simDim);
                            bufferCellSizeOpenPMD = std::make_shared<pmacc::HostDeviceBuffer<float_X, 1>>(simDim);

                            auto dataBoxExtent = bufferExtentOpenPMD->getHostBuffer().getDataBox();
                            auto dataBoxCellSize = bufferCellSizeOpenPMD->getHostBuffer().getDataBox();

                            for(uint32_t d = 0u; d < simDim; d++)
                            {
                                extentOpenPMD[aligningAxisIndex[d]] = extentRaw[d]; // axis alignment and type conversion (openPMD::Extent to DataSpace<simDim>)
                                dataBoxExtent(aligningAxisIndex[d]) = static_cast<float_X>(extentRaw[d]);
                                cellSizeOpenPMD[aligningAxisIndex[d]] = cellSizeRaw[d] * meshE.gridUnitSI() / UNIT_LENGTH;  // axis alignment and grid spaing in unit length, brauch ich das überhaupt?
                                dataBoxCellSize(aligningAxisIndex[d]) = cellSizeRaw[d] * meshE.gridUnitSI() / UNIT_LENGTH;
                            }

                            // copy attribute data to device
                            bufferExtentOpenPMD->hostToDevice();
                            bufferCellSizeOpenPMD->hostToDevice();
                            eventSystem::getTransactionEvent().waitForFinished();  //brauche ich das auch nach dem 1. ?
                            log<picLog::PHYSICS>("HostToDevice attribute data successful");

                            // besser: DataType abfragen!
                            using dataType = float_64; // meshRecordE.getDatatype(); error: is not a type name

                            fieldBufferE = std::make_shared<pmacc::HostDeviceBuffer<dataType, simDim>>(extentOpenPMD);  // oben instanziiert
                            //dc.share(fieldBufferE);  // brauche ich das hier?
                            // error: no suitable user-defined conversion

                            auto dataEx = std::shared_ptr<dataType>{nullptr};
                            dataEx = meshRecordE.loadChunk<dataType>();

                            // This is MPI collective and so has to be done by all ranks
                            series.flush();

                            // test print
                            for( size_t col = 0;
                                col < extentRaw[1] && col < 5;
                                ++col )
                                log<picLog::PHYSICS>("Field data %1%") % dataEx.get()[col];

                            auto const numElements = std::accumulate(
                            std::begin(extentRaw),
                            std::end(extentRaw),
                            1u,
                            std::multiplies<uint32_t>());

                            auto hostEDataBox = fieldBufferE->getHostBuffer().getDataBox();

                            for(uint32_t linearIdx = 0u; linearIdx < numElements; linearIdx++)
                            {
                                // .get() returns a flat vector, has to be reshaped to fill dataBoxes
                                DataSpace<simDim> openPMDIdx;  // DataSpace
                                auto tmpIndex = linearIdx;
                                for(int32_t d = simDim - 1; d >= 0; d--)
                                {
                                    openPMDIdx[aligningAxisIndex[d]] = tmpIndex % extentRaw[d];
                                    tmpIndex /= extentRaw[d];
                                }
                                hostEDataBox(openPMDIdx) = dataEx.get()[linearIdx] * meshRecordE.unitSI() / UNIT_EFIELD;
                            }

                            // Copy host data to the device
                            fieldBufferE->hostToDevice();
                            eventSystem::getTransactionEvent().waitForFinished();
                            log<picLog::PHYSICS>("HostToDevice fielddata successful");
                            // series.close();  // Notwendig? Error: has no member close
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
                        // can be static?
                        pmacc::DataSpace<simDim> extentPIC = subGrid.getGlobalDomain().size;  // global extent, No of cells
                        

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE InsightPulseFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
                            : timePIC(currentStep * DELTA_T)
                        {
                            log<picLog::PHYSICS>("currentStep %1%") % currentStep;
                            auto& openPMDdata = OpenPMDdata<T_Params>::get();
                            log<picLog::PHYSICS>("Got OpenPMDdata");


                            devDataBox = openPMDdata.fieldBufferE->getDeviceBuffer().getDataBox();
                            log<picLog::PHYSICS>("Got DeviceDataBox");

                            // OpenPMD attributes
                            extentOpenPMDdataBox = openPMDdata.bufferExtentOpenPMD->getDeviceBuffer().getDataBox();
                            cellSizeOpenPMDdataBox = openPMDdata.bufferCellSizeOpenPMD->getDeviceBuffer().getDataBox();
                        }

                        // Folgende Funktionen gibt es auch in BaseFinctor, evtl den nutzen!
                        /** Read incident field E value for the given position and time step
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @return incident field E value in internal units
                         */
                        HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
                        {
                            return getPolarisationVector() * getValueE(totalCellIdx);
                        }

                        //! Get a unit vector with linear E polarization
                        HDINLINE static constexpr float3_X getPolarisationVector()
                        {
                            return float3_X(Unitless::POL_DIR_X, Unitless::POL_DIR_Y, Unitless::POL_DIR_Z);
                        }

                        //! Get a normalized 3-dimensional direction vector
                        // needed for ApproximateIncidentB
                        // kommt eigentlich aus Base!
                        HDINLINE static float3_X getDirection()
                        {
                            return float3_X(Unitless::DIR_X, Unitless::DIR_Y, Unitless::DIR_Z);
                        }

                        /** Linear interpolation of scalar field data stored on a regular grid to PIConGPU-intern grid
                         * Extrapolation to default values
                         * Kann man im Idealfall für B wiederverwenden
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         */
                        HDINLINE float_X GridInterpolator(floatD_X const& totalCellIdx) const
                        {
                            
                            
                            
                            // evtl schon im Konstruktor initialisieren
                            auto const posPIC = totalCellIdx * cellSize;   // point at which to evaluate field data
                            // evtl schon im Konstruktor initialisieren (als static?)
                            // 3er vektor als int?
                            int const propAxisIdx = static_cast<int>(pmacc::math::dot(float3_X{0.0_X, 1.0_X, 2.0_X}, getDirection()));  // 0 = x, 1 = y, 2 = z


                            // centering OpenPMD data in incident plane
                            // offset of OpenPMD data w.r.t. Origin in Simulation
                            // Sinnvoll nur für transversale Richtung
                            // evtl schon im Konstruktor initialisieren (als static?), dann reicht dataBox auch hostseitig
                            floatD_X offsetOpenPMD; // const? und ist es vllt schneller, extent und CellSize in float3_X zu casten?
                            for(uint32_t d = 0u; d < simDim; d++)
                            {
                                offsetOpenPMD[d] = (static_cast<float_X>(extentPIC[d]) * cellSize[d]
                                    - extentOpenPMDdataBox(d) * cellSizeOpenPMDdataBox(d)) / 2.0_X;  // auch -1?
                                    // was ist wenn Offset negativ ist? --> throw error: transversal window too small
                                    // oder daten croppen?
                                if(d != propAxisIdx)
                                    PMACC_DEVICE_VERIFY_MSG(offsetOpenPMD[d] >= 0, "Error: transversal simulation window is too small to depict all data. Please choose a bigger one");
                            }
                            
                            // if posPIC lies outside of stored field data, return default value                            
                            DataSpace<simDim> idxClosest; // = uint vector?
                            float3_X idxClosestRaw; 

                            for(uint32_t d = 0u; d < simDim; d++)
                            {
                                //PMACC_DEVICE_VERIFY_MSG(0 == 1, "extentOpenPMDdataBox[%u] = %f", d, extentOpenPMDdataBox(d));
                                if(d != propAxisIdx and
                                    (posPIC[d] < offsetOpenPMD[d] or posPIC[d] > offsetOpenPMD[d] + (extentOpenPMDdataBox(d)-1.0_X) * cellSizeOpenPMDdataBox(d))  // transversal axes
                                    or timePIC < 0 or timePIC > (extentOpenPMDdataBox(d)-1.0_X) * cellSizeOpenPMDdataBox(d) / SPEED_OF_LIGHT)  // OpenPMD propagation axis = time axis
                                    return Unitless::defaultEFieldValue;
                                // is SPEED_OF_LIGHT in internal units?
                                // maybe do time already in hosttodevice
                                else
                                {
                                    idxClosestRaw[d] = (posPIC[d] - offsetOpenPMD[d]) / cellSizeOpenPMDdataBox(d);  // transversal / spatial
                                    idxClosest[d] = static_cast<int>(idxClosestRaw[d] + 0.5_X);  // geht das auch mit static_cast auf alles am Ende / broadcasting?
                                }
                            }
                            // longitudinal
                            idxClosestRaw[propAxisIdx] = extentOpenPMDdataBox(propAxisIdx)-1.0_X - timePIC / cellSizeOpenPMDdataBox(propAxisIdx) * SPEED_OF_LIGHT;
                            idxClosest[propAxisIdx] = static_cast<int>(idxClosestRaw[propAxisIdx] + 0.5);

                            for(uint32_t d = 0u; d < simDim; d++)
                            {
                                PMACC_DEVICE_VERIFY_MSG(idxClosest[d] >= 0, "Error: idxClosest[%u] < 0 ", d);
                                PMACC_DEVICE_VERIFY_MSG(idxClosest[d] <= extentOpenPMDdataBox(d) - 1, "Error: idxClosest%u] > extentOpenPMDdataBox(d) - 1", d);
                            }
                            
                            float_X interpE; 
                            // find the other 7 nearest neighbours
                            // shift to next to nearest neighbour index
                            floatD_X idxShift, weight;  // eigentlich int, aber ich brauche die invertierung
                            for(uint32_t d = 0u; d < simDim; d++)
                            {
                                if(idxClosestRaw[d] - static_cast<float_X>(idxClosest[d]) < 0)
                                    idxShift[d] = -1;
                                else
                                    idxShift[d] = 1;
                                    
                                weight[d] = pmacc::math::abs(static_cast<float_X>(idxClosest[d]) - idxClosestRaw[d]);
                            }
                            
                            interpE = static_cast<float_X>(devDataBox(idxClosest)) * weight.productOfComponents();

                            return interpE; //static_cast<float_X>(devDataBox(idxClosest));
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
                        //typename pmacc::Buffer<float_64, simDim>::DataBoxType devDataBox;
                        PMACC_ALIGN(devDataBox, typename pmacc::Buffer<float_64, simDim>::DataBoxType);
                        typename pmacc::Buffer<float_X, 1>::DataBoxType extentOpenPMDdataBox;
                        typename pmacc::Buffer<float_X, 1>::DataBoxType cellSizeOpenPMDdataBox;

                    }; // InsightPulseFunctorIncidentE
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

                /** Get type of incident field B functor for the Insight laser profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentB<profiles::InsightPulse<T_Params>>
                {
                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::InsightPulse<T_Params>>::type>;
                };

            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
