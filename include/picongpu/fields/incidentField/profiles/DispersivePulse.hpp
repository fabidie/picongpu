/* Copyright 2022 Fabia Dietrich, Klaus Steiniger, Richard Pausch
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
#include "pmacc/memory/buffers/HostDeviceBuffer.hpp"

#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/math/Complex.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include <iostream>
#include <fstream>


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
                    /** Unitless DispersivePulse parameters
                     *
                     * @tparam T_Params user (SI) parameters
                     */
                    template<typename T_Params>
                    struct DispersivePulseUnitless : public BaseParamUnitless<T_Params>
                    {
                        /*
                         * Probleme mit:
                         * 
                         * Währent Init-Time: massive Feldproduktion im origin (0, 0, 0)
                         * Auswahl y/z index prüfen
                         * Polarisationsrichtung: y scheint nicht zu passen ...
                         * 
                         * Problem: 512 * dz als Simulationsvolumen, Array deckt aber nur die Hälfte ab
                         * Nur auf 1 GPU laufen lassen --> hilft nicht
                         * 
                         * LINEARER ZUSAMMENHANG ZWISCHEN SIMULATIONSVOLUMEN UND ARRAYGRÖßE:
                         * 512 --> size(z) = 262
                         * 416 --> size(z) = 166
                         * 320 --> size(z) = 70
                         * 256 --> size(z) = 5
                         * 224 --> error (segmentation fault) --> probably no entries at all...
                         * 128 --> error (segmentation fault)
                         *                          
                         * Fokus position prüfen
                         *                          
                         */
                        
                        
                        
                        //! User SI parameters
                        using Params = T_Params;

                        //! Base unitless parameters
                        using Base = BaseParamUnitless<T_Params>;

                        // unit: UNIT_LENGTH
                        static constexpr float_X W0 = static_cast<float_X>(Params::W0_SI / UNIT_LENGTH);
                        

                        // rayleigh length in propagation direction
                        static constexpr float_X rayleighLength
                            = pmacc::math::Pi<float_X>::value * W0 * W0 / Base::WAVE_LENGTH;

                        // unit: UNIT_TIME
                        // corresponds to period length of DFT
                        static constexpr float_X INIT_TIME
                            = static_cast<float_X>(Params::PULSE_INIT) * Base::PULSE_DURATION;

                        // Dispersion parameters
                        // unit: UNIT_LENGTH * UNIT_TIME
                        static constexpr float_X SD = static_cast<float_X>(Params::SD_SI / UNIT_TIME / UNIT_LENGTH);
                        // unit: rad * UNIT_TIME
                        static constexpr float_X AD = static_cast<float_X>(Params::AD_SI / UNIT_TIME);
                        // unit: UNIT_TIME^2
                        static constexpr float_X GDD = static_cast<float_X>(Params::GDD_SI / UNIT_TIME / UNIT_TIME);
                        // unit: UNIT_TIME^3
                        static constexpr float_X TOD
                            = static_cast<float_X>(Params::TOD_SI / UNIT_TIME / UNIT_TIME / UNIT_TIME);
                    };

                    /** DispersivePulse incident E functor
                     * 
                     * Test function!
                     * works just if the user sets x as propagation direction and y as polarisation direction
                     *
                     * @tparam T_Params parameters
                     */
                    template<typename T_Params>
                    struct DispersivePulseFunctorIncidentE
                        : public DispersivePulseUnitless<T_Params>
                        , public incidentField::detail::BaseFunctorE<T_Params>
                    {
                        //! Unitless parameters type
                        using Unitless = DispersivePulseUnitless<T_Params>;

                        //! Base functor type
                        using Base = incidentField::detail::BaseFunctorE<T_Params>;
                        
                        // complex data type
                        using complex_X = alpaka::Complex<float_X>;
                        
                        // Data Box types for complex field values
                        typename HostDeviceBuffer<complex_X, DIM3>::DataBoxType EwBox;
                        // Data Box types for transversal coordinates and frequency array
                        typename HostDeviceBuffer<float_X, DIM1>::DataBoxType YBox;
                        typename HostDeviceBuffer<float_X, DIM1>::DataBoxType ZBox;
                        typename HostDeviceBuffer<float_X, DIM1>::DataBoxType WBox;
                        
                        // calculate the sizes of the data boxes
                        // Cell sizes in UNIT_LENGTH
                        float_X const dx = static_cast<float_X>(picongpu::SI::CELL_WIDTH_SI / UNIT_LENGTH);
                        float_X const dy = static_cast<float_X>(picongpu::SI::CELL_HEIGHT_SI / UNIT_LENGTH);
                        float_X const dz = static_cast<float_X>(picongpu::SI::CELL_DEPTH_SI / UNIT_LENGTH);
                        // global simulation volume in internal length unit
                        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                        const DataSpace<simDim> globalSize = subGrid.getGlobalDomain().size; 
                        float3_X gridSizeTransformed
                            = this->getInternalCoordinates(precisionCast<float_X>(globalSize));
                        // transversal coordinate boxes size
                        const int ySamples = static_cast<int>(gridSizeTransformed.y() / dy ); // static?
                        const int zSamples = static_cast<int>(gridSizeTransformed.z() / dz );
                        // Omega Box size
                        float_X const dt = static_cast<float_X>(picongpu::SI::DELTA_T_SI / UNIT_TIME);
                        float_X N_raw = Unitless::INIT_TIME / dt;
                        const int wSamples = static_cast<int>(N_raw * 0.5_X);  // = n
                        

                        /** Create a functor on the host side for the given time step
                         * 
                         * loading E(y, z, Omega)- data from Device into an array
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE DispersivePulseFunctorIncidentE(float_X const currentStep, float3_64 const unitField)
                            : Base(currentStep, unitField)
                        {
                            // y, z are the transversal directions and x is the propagation direction
                            // YBox contains transversal y coordinates
                            // internal units
                            static auto bufferY = loadYData();
                            YBox = bufferY->getDeviceBuffer().getDataBox();
                            
                            // ZBox contains transversal z coordinates
                            // internal units
                            static auto bufferZ = loadZData();
                            ZBox = bufferZ->getDeviceBuffer().getDataBox();
                            
                            // Omega array
                            static auto bufferW = loadWData();
                            WBox = bufferW->getDeviceBuffer().getDataBox();
                            
							// E(x, y, z, w)-array
							static auto bufferEw = loadEwData();
                            EwBox = bufferEw->getDeviceBuffer().getDataBox();
                        }
                        
                        auto loadYData()
                        {
                            auto bufferY = std::make_unique<HostDeviceBuffer<float_X, DIM1>>(ySamples);
                            auto hostDBox = bufferY->getHostBuffer().getDataBox();
                            for(int y=0; y<ySamples; ++y)
                            {
                                // evenly spaced sample points in internal coordinates
                                hostDBox[y] = static_cast<float_X>(y) * dy;
							}
                            bufferY->hostToDevice();
                            return bufferY;
                        }
                        
                        auto loadZData()
                        {
                            auto bufferZ = std::make_unique<HostDeviceBuffer<float_X, DIM1>>(zSamples);
                            auto hostDBox = bufferZ->getHostBuffer().getDataBox();
                            std::ofstream myfile;
                            myfile.open ("zData.txt");
                            for(int z=0; z<zSamples; ++z)
                            {
                                // evenly spaced sample points in internal coordinates
                                hostDBox[z] = static_cast<float_X>(z) * dz;
                                
                                myfile << static_cast<float_X>(z) * dz * UNIT_LENGTH << '\n';
							}
					
					        myfile.close();
                            
                            bufferZ->hostToDevice();
                            return bufferZ;
                        }
                        
                        auto loadWData()
                        {
							// zero-padding of INSIGHT-data
							
							auto bufferW = std::make_unique<HostDeviceBuffer<float_X, DIM1>>(wSamples + 1);
                            auto hostDBox = bufferW->getHostBuffer().getDataBox();
							
							for(int k=0; k<=wSamples; k++)
                            {
								// starting at Omega = 0
								// Omega = 2pi*k/T
                                hostDBox[k] = static_cast<float_X>(k) * pmacc::math::Pi<float_X>::doubleValue
                                    / Unitless::INIT_TIME;
							}
							bufferW->hostToDevice();
							return bufferW;
						}
                        
                        auto loadEwData()
                        {
                            auto bufferEw = std::make_unique<HostDeviceBuffer<complex_X, DIM3>>(DataSpace<DIM3>(
                                ySamples, zSamples, wSamples+1));
                            auto hostDBox = bufferEw->getHostBuffer().getDataBox();
                            
                            std::ofstream Ewfile;
                            Ewfile.open ("EwData.txt");
                            Ewfile << ySamples << ' ' << zSamples << ' ' << wSamples << '\n';

                            for(int y = 0; y < ySamples; y++)
                            {
                                for(int z = 0; z < zSamples; z++)
                                {
                                    for(int f = 0; f <= wSamples; f++)
                                    {
										// here: position of incident plane x set to 0
                                        // Ew(Omega=0) = 0 has to be set (not calculated!)
                                        // Omega Array is one entry longer than 3rd dimension of Ew!
                                        // hostDBox(DataSpace<DIM3>(y,z,f)) = 
                                        float_X Omega = static_cast<float_X>(f) * pmacc::math::Pi<float_X>::doubleValue / Unitless::INIT_TIME;
                                        complex_X EwData = complex_X(
                                            pmacc::math::euler(amp(16.0_X * dx, static_cast<float_X>(y)*dy, static_cast<float_X>(z)*dz, Omega), 
										                      -phi(16.0_X * dx, static_cast<float_X>(y)*dy, static_cast<float_X>(z)*dz, Omega)));
                                        hostDBox[y][z][f] = EwData;
										                      
										Ewfile << EwData.real()*EwData.real() + EwData.imag()* EwData.imag() << ' ';
									}
								}
							}
							
							Ewfile.close();

                            bufferEw->hostToDevice();
                            return bufferEw;
                        }

                        /** Calculate incident field E value for the given position
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @return incident field E value in internal units
                         */
                        HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
                        {
                            if constexpr(Unitless::Polarisation == PolarisationType::Linear)
                                return this->getLinearPolarizationVector() * getValueE(totalCellIdx, 0.0_X);
                            else
                            {
                                // macht Polarisierung hier Sinn mit den Messwerten?
                                auto const phaseShift = pmacc::math::Pi<float_X>::halfValue;
                                return this->getCircularPolarizationVector1() * getValueE(totalCellIdx, phaseShift)
                                    + this->getCircularPolarizationVector2() * getValueE(totalCellIdx, 0.0_X);
                            }
                        }

                        /** Helper function to calculate the electric field in frequency domain.
                         * Initial frequency dependent complex phase expanded up to third order in (Omega - Omega_0).
                         * Takes only first order angular dispersion d theta / d Omega = theta^prime into account
                         * and neglects all higher order angular dispersion terms, e.g. theta^{prime prime},
                         * theta^{prime prime prime}, ...
                         *
                         * @param Omega frequency for which the E-value is calculated
                         */
                        HDINLINE float_X expandedWaveVectorX(float_X const Omega) const
                        {
                            return Unitless::W0 / SPEED_OF_LIGHT
                                * (Unitless::w * Unitless::AD * (Omega - Unitless::w)
                                   + Unitless::AD * (Omega - Unitless::w) * (Omega - Unitless::w)
                                   - Unitless::w / 6.0_X * Unitless::AD * Unitless::AD * Unitless::AD
                                       * (Omega - Unitless::w) * (Omega - Unitless::w) * (Omega - Unitless::w));
                        }

                        /** The following two functions provide the electric field in frequency domain
                         * E(Omega) = amp * exp(-i*phi)
                         * Please ensure that E(Omega = 0) = 0 (no constant field contribution), i.e. the pulse
                         * length has to be big enough. Otherwise the implemented DFT will produce wrong results.
                         *
                         * @param x position of incident plane
                         * @param y, z transversal direction position
                         * @param Omega frequency for which the E-value is calculated
                         */
                        HDINLINE float_X amp(float_X const x, float_X const y, float_X const z, float_X const Omega) const
                        {

                            // calculate focus position relative to the current point in the propagation direction
                            //auto const focusRelativeToOrigin = this->focus - this->origin;
                            // current distance to focus position
                            //float_X const focusPos = math::sqrt(pmacc::math::l2norm2(focusRelativeToOrigin)) - x;
                            float_X const focusPos = (this->focus)[0] - 16.0_X*dx;
                            // beam waist at the generation plane so that at focus we will get W0
                            float_X const waist = Unitless::W0
                                * math::sqrt(1.0_X
                                             + (focusPos / Unitless::rayleighLength)
                                                 * (focusPos / Unitless::rayleighLength));

                            // Initial frequency dependent complex phase
                            float_X alpha = expandedWaveVectorX(Omega);

                            // Center of a frequency's spatial distribution
                            float_X center = Unitless::SD * (Omega - Unitless::w)
                                + SPEED_OF_LIGHT * alpha * focusPos / (Unitless::W0 * Unitless::w);

                            // gaussian envelope in frequency domain
                            float_X mag = math::exp(
                                -(Omega - Unitless::w) * (Omega - Unitless::w) * Unitless::PULSE_DURATION
                                * Unitless::PULSE_DURATION);

                            // transversal envelope
                            mag *= math::exp(
                                -(y - center) * (y - center) / waist / waist); // envelope y - direction
                            mag *= math::exp(
                                -(z - center) * (z - center) / waist / waist); // envelope z - direction

                            // distinguish between dimensions
                            // check dimensionality!
                            if constexpr(simDim == DIM2)  
                            {
                                // pos has just two entries: pos[0] as propagation direction and pos[1] as transversal
                                // direction
                                mag *= math::sqrt(Unitless::W0 / waist);
                            }
                            else if constexpr(simDim == DIM3)
                            {
                                mag *= Unitless::W0 / waist;
                            }

                            // Normalization to Amplitude
                            mag *= math::sqrt(pmacc::math::Pi<float_X>::doubleValue * 0.5_X) * 2.0_X
                                * Unitless::PULSE_DURATION * Unitless::AMPLITUDE;

                            // Dividing amplitude by 2 to compensate doubled spectral field strength
                            // resulting from E(-Omega) = E*(Omega), which has to be fulfilled for E(t) to be real
                            mag *= 0.5_X;

                            return mag;
                        }

                        HDINLINE float_X phi(float_X const x, float_X const y, float_X const z, float_X const Omega) const
                        {
                            // calculate focus position relative to the current point in the propagation direction
                            auto const focusRelativeToOrigin = this->focus - this->origin;
                            float_X const focusPos = math::sqrt(pmacc::math::l2norm2(focusRelativeToOrigin)) - x;

                            // Initial frequency dependent complex phase
                            float_X alpha = expandedWaveVectorX(Omega);

                            // Center of a frequency's spatial distribution
                            float_X center = Unitless::SD * (Omega - Unitless::w)
                                + SPEED_OF_LIGHT * alpha * focusPos / (Unitless::W0 * Unitless::w);

                            // inverse radius of curvature of the pulse's wavefronts
                            auto const R_inv = -focusPos
                                / (Unitless::rayleighLength * Unitless::rayleighLength + focusPos * focusPos);
                            // the Gouy phase shift
                            auto const xi = math::atan(-focusPos / Unitless::rayleighLength);

                            // shifting pulse for half of INIT_TIME to start with the front of the laser pulse
                            constexpr auto mue = 0.5_X * Unitless::INIT_TIME;
                            float_X const timeDelay = mue + focusPos / SPEED_OF_LIGHT;

                            float_X phase = -Omega * focusPos / SPEED_OF_LIGHT
                                + 0.5_X * Unitless::GDD * (Omega - Unitless::w) * (Omega - Unitless::w)
                                + Unitless::TOD / 6.0_X * (Omega - Unitless::w) * (Omega - Unitless::w)
                                    * (Omega - Unitless::w)
                                 + Unitless::LASER_PHASE + Omega * timeDelay;

                            phase += ((y - center) * (y - center) + (z - center) * (z - center))
                                * Omega * 0.5_X * R_inv / SPEED_OF_LIGHT;
                            phase -= alpha * (y + z) / Unitless::W0;

                            // distinguish between dimensions
                            if constexpr(simDim == DIM2)
                            {
                                phase += alpha * alpha / 4.0_X * focusPos / Unitless::rayleighLength;
                                phase -= 0.5_X * xi;
                            }
                            else if constexpr(simDim == DIM3)
                            {
                                phase += alpha * alpha / 2.0_X * focusPos / Unitless::rayleighLength;
                                phase -= xi;
                            }
                            return phase;
                        }

                    private:
                        HDINLINE int findClosestIdx(HostDeviceBuffer<float_X, DIM1>::DataBoxType const& arr, int n, float_X target) const
                        {
                            int left = 0, right = n - 1;
                            while (left < right) {
                                if (((arr[left] - target)*(arr[left] - target)) <= ((arr[right] - target)*(arr[right] - target))) {
                                    right--;
                                }
                                else {
                                    left++;
                                }
                            }
                            return left;
                        }
                        
                        /** Get value of E field in time domain for the given position, using DFT
                         * Interpolation order of DFT given via timestep in grid.param and INIT_TIME
                         * Neglecting the constant part of DFT (k = 0) because there should be no constant field
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @param phaseShift additional phase shift to add on top of everything else,
                         *                   in radian
                         */
                        HDINLINE float_X getValueE(floatD_X const& totalCellIdx, float_X const phaseShift) const
                        {   
                            float3_X pos = this->getInternalCoordinates(totalCellIdx);
                                                     
                            auto const time = this->getCurrentTime(totalCellIdx);
                            if(time < 0.0_X)
                                return 0.0_X;
                                
                            // turning Laser off after producing one pulse (during INIT_TIME) to avoid periodic pulses
                            else if(time > Unitless::INIT_TIME)
                                return 0.0_X;
                            
                            int y = findClosestIdx(YBox, ySamples, static_cast<float_X>(pos[1]));
                            int z = findClosestIdx(ZBox, zSamples, static_cast<float_X>(pos[2]));
                            
                            
                            float_X E_t = 0.0_X; // electric field in time-domain
                            for(int k = 1; k <= wSamples; k++)
                            {
                                // error: no instance of constructor "Norm" matches the argument list
                                float_X abs = math::sqrt(EwBox[y][z][k].real()*EwBox[y][z][k].real() + EwBox[y][z][k].imag()*EwBox[y][z][k].imag());
                                float_X arg = std::atan2(EwBox[y][z][k].imag(), EwBox[y][z][k].real());
                                
                                E_t += 2.0_X * abs * math::cos(arg + phaseShift)
                                         * math::cos(WBox[k] * time)
                                     + 2.0_X * abs * math::sin(arg + phaseShift)
                                         * math::sin(WBox[k] * time);
                       
                            }

                            E_t /= static_cast<float_X>(2 * wSamples + 1) * dt; // Normalization from DFT
                            
                            return E_t;
                        } // getValueE
                    }; // DispersivePulseFunctorIncidentE

                    /** DispersivePulse incident B functor
                     *
                     * EXPERIMENTAL
                     * Do not use for production! Refactoring required
                     *
                     * @tparam T_Params parameters
                     */
                    template<typename T_Params>
                    struct DispersivePulseFunctorIncidentB : public DispersivePulseFunctorIncidentE<T_Params>
                    {
                        //! E Functor type
                        using EFunctor = DispersivePulseFunctorIncidentE<T_Params>;

                        //! Unitless parameters type
                        using Unitless = DispersivePulseUnitless<T_Params>;

                        //! Relation between unitField for E and B: E = B * unitConversionBtoE
                        static constexpr float_64 unitConversionBtoE = UNIT_EFIELD / UNIT_BFIELD;

                        using complex_X = alpaka::Complex<float_X>;

                        //! Cell size in UNIT_LENGTH
                        // not permutated! -> probably cause of bug
                        float_X const d0 = static_cast<float_X>(picongpu::SI::CELL_WIDTH_SI / UNIT_LENGTH);
                        float_X const d1 = static_cast<float_X>(picongpu::SI::CELL_HEIGHT_SI / UNIT_LENGTH);
                        float_X const d2 = static_cast<float_X>(picongpu::SI::CELL_DEPTH_SI / UNIT_LENGTH);
                        float3_X const d = float3_X(d0, d1, d2);

                        /** Create a functor on the host side for the given time step
                         *
                         * @param currentStep current time step index, note that it is fractional
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldB_internal = fieldB_SI / unitField
                         */
                        HINLINE DispersivePulseFunctorIncidentB(float_X const currentStep, float3_64 const unitField)
                            : EFunctor(currentStep, unitField * unitConversionBtoE)
                        {
                        }

                        /** Calculate incident field B value for the given position
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @return incident field B value in internal units
                         */
                        HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
                        {
                            return float3_X(
                                getValueB(0, totalCellIdx),
                                getValueB(1, totalCellIdx),
                                getValueB(2, totalCellIdx));
                        }

                        /** Calculate incident field B in fourier space for the given position, solving
                         *      B_Omega = i / Omega * rot(E_Omega)
                         * by discretizing it and using the Finite-Difference method
                         * (https://picongpu.readthedocs.io/en/latest/models/AOFDTD.html)
                         *
                         * @param axis (external) direction of returned B value (0 for x, 1 for y, 2 for z)
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @param Omega angular frequency for which the corresponding B-value is wanted
                         * @return incident field B value in fourier space in unit size
                         */
                        HDINLINE complex_X
                        B_Omega(int const axis, floatD_X const& totalCellIdx, float_X const Omega) const
                        {
                            int i, j; // for E field direction and differentiation direction
                            complex_X diEj, djEi; // rot E

                            if(axis == 0)
                            {
                                i = 1;
                                j = 2;
                            }
                            else if(axis == 1)
                            {
                                i = 2;
                                j = 0;
                            }
                            else if(axis == 2)
                            {
                                i = 0;
                                j = 1;
                            }

                            float3_X iIdxForw = float3_X(totalCellIdx[0], totalCellIdx[1], totalCellIdx[2]);
                            iIdxForw[i] = totalCellIdx[i] + 0.5_X;
                            float3_X iIdxBackw = float3_X(totalCellIdx[0], totalCellIdx[1], totalCellIdx[2]);
                            iIdxBackw[i] = totalCellIdx[i] - 0.5_X;
                            float3_X jIdxForw = float3_X(totalCellIdx[0], totalCellIdx[1], totalCellIdx[2]);
                            jIdxForw[j] = totalCellIdx[j] + 0.5_X;
                            float3_X jIdxBackw = float3_X(totalCellIdx[0], totalCellIdx[1], totalCellIdx[2]);
                            jIdxBackw[j] = totalCellIdx[j] - 0.5_X;

                            if constexpr(Unitless::Polarisation == PolarisationType::Linear)
                            {
                                // using euler function to convert (amp, phase) into (real, imag)
                                diEj = (pmacc::math::euler(
                                            EFunctor::amp(iIdxForw, Omega),
                                            -EFunctor::phi(iIdxForw, Omega, 0.0_X))
                                        - pmacc::math::euler(
                                            EFunctor::amp(iIdxBackw, Omega),
                                            -EFunctor::phi(iIdxBackw, Omega, 0.0_X)))
                                    * this->getLinearPolarizationVector()[j] / d[i];

                                djEi = (pmacc::math::euler(
                                            EFunctor::amp(jIdxForw, Omega),
                                            -EFunctor::phi(jIdxForw, Omega, 0.0_X))
                                        - pmacc::math::euler(
                                            EFunctor::amp(jIdxBackw, Omega),
                                            -EFunctor::phi(jIdxBackw, Omega, 0.0_X)))
                                    * this->getLinearPolarizationVector()[i] / d[j];
                            } // Linear

                            else
                            {
                                auto const phaseShift = pmacc::math::Pi<float_X>::halfValue;
                                diEj = (pmacc::math::euler(
                                            EFunctor::amp(iIdxForw, Omega),
                                            -EFunctor::phi(iIdxForw, Omega, phaseShift))
                                        - pmacc::math::euler(
                                            EFunctor::amp(iIdxBackw, Omega),
                                            -EFunctor::phi(iIdxBackw, Omega, phaseShift)))
                                        * this->getCircularPolarizationVector1()[j] / d[i]
                                    + (pmacc::math::euler(
                                           EFunctor::amp(iIdxForw, Omega),
                                           -EFunctor::phi(iIdxForw, Omega, 0.0_X))
                                       - pmacc::math::euler(
                                           EFunctor::amp(iIdxBackw, Omega),
                                           -EFunctor::phi(iIdxBackw, Omega, 0.0_X)))
                                        * this->getCircularPolarizationVector2()[j] / d[i];

                                djEi = (pmacc::math::euler(
                                            EFunctor::amp(jIdxForw, Omega),
                                            -EFunctor::phi(jIdxForw, Omega, phaseShift))
                                        - pmacc::math::euler(
                                            EFunctor::amp(jIdxBackw, Omega),
                                            -EFunctor::phi(jIdxBackw, Omega, phaseShift)))
                                        * this->getCircularPolarizationVector1()[i] / d[j]
                                    + (pmacc::math::euler(
                                           EFunctor::amp(jIdxForw, Omega),
                                           -EFunctor::phi(jIdxForw, Omega, 0.0_X))
                                       - pmacc::math::euler(
                                           EFunctor::amp(jIdxBackw, Omega),
                                           -EFunctor::phi(jIdxBackw, Omega, 0.0_X)))
                                        * this->getCircularPolarizationVector2()[i] / d[j];
                            } // Circular

                            return complex_X(0, 1) * (diEj - djEi) / Omega;

                        } // B_Omega

                    private:
                        /** Get value of B field in time domain for the given position, using DFT
                         * Interpolation order of DFT given via timestep in grid.param and INIT_TIME
                         *
                         * The constant part of the FT (k = 0) is neglected because there should be no
                         * constant field.
                         *
                         * @param totalCellIdx cell index in the total domain (including all moving window slides)
                         * @param axis direction of returned B value (0 for x, 1 for y, 2 for z)
                         * @return incident field B value in unit size
                         */
                        HDINLINE float_X getValueB(int const axis, floatD_X const& totalCellIdx) const
                        {
                            auto const time = this->getCurrentTime(totalCellIdx);
                            if(time < 0.0_X)
                                return 0.0_X;

                            // turning Laser off after producing one pulse (during INIT_TIME) to avoid periodic pulses
                            else if(time > Unitless::INIT_TIME)
                                return 0.0_X;

                            // timestep also in UNIT_TIME
                            float_X const dt = static_cast<float_X>(picongpu::SI::DELTA_T_SI / UNIT_TIME);
                            // interpolation order
                            float_X N_raw = Unitless::INIT_TIME / dt;
                            int n = static_cast<int>(N_raw / 2); // -0 instead of -1 for rounding up N_raw

                            float_X B_t = 0.0_X; // magnetic field in time-domain
                            float_X Omk = 0.0_X; // stores angular frequency for DFT-loop; Omk= 2pi*k/T

                            for(int k = 1; k < n + 1; k++)
                            {
                                Omk = static_cast<float_X>(k) * pmacc::math::Pi<float_X>::doubleValue
                                    / Unitless::INIT_TIME;
                                B_t += 2.0_X * B_Omega(axis, totalCellIdx, Omk).real() / dt * math::cos(Omk * time)
                                    - 2.0_X * B_Omega(axis, totalCellIdx, Omk).imag() / dt * math::sin(Omk * time);
                            }

                            B_t /= static_cast<float_X>(2 * n + 1);

                            return B_t;
                        } // getValueB
                    }; // DispersivePulseFunctorIncidentB

                } // namespace detail

                template<typename T_Params>
                struct DispersivePulse
                {
                    //! Get text name of the incident field profile
                    static HINLINE std::string getName()
                    {
                        return "DispersivePulse";
                    }
                };

            } // namespace profiles

            namespace detail
            {
                /** Get type of incident field E functor for the dispersive laser profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentE<profiles::DispersivePulse<T_Params>>
                {
                    using type = profiles::detail::DispersivePulseFunctorIncidentE<T_Params>;
                };

                /** Get type of incident field B functor for the dispersive laser profile type
                 *
                 * @tparam T_Params parameters
                 */
                template<typename T_Params>
                struct GetFunctorIncidentB<profiles::DispersivePulse<T_Params>>
                {
                    // EXPERIMENTAL - Do NOT use!
                    // using type = profiles::detail::DispersivePulseFunctorIncidentB<T_Params>;

                    using type = detail::ApproximateIncidentB<
                        typename GetFunctorIncidentE<profiles::DispersivePulse<T_Params>>::type>;
                };

            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
