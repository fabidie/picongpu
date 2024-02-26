/* Copyright 2024 Julian Lenz
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

#include "pmacc/math/Vector.hpp"
#include "pmacc/meta/ForEach.hpp"
#include "pmacc/meta/conversion/MakeSeq.hpp"

#include <boost/mp11/bind.hpp>

#include <numeric>
#include <type_traits>

#include <nlohmann/json.hpp>


namespace picongpu
{
    namespace traits
    {
        template<typename, typename = void>
        inline constexpr bool providesMetadata = false;

        template<typename T>
        inline constexpr bool providesMetadata<T, std::void_t<decltype(std::declval<T>().metadata())>> = true;

        template<typename, typename = void>
        inline constexpr bool providesMetadataAtCT = false;

        template<typename T>
        inline constexpr bool providesMetadataAtCT<T, std::void_t<decltype(T::metadata())>> = true;

        template<typename, typename = void>
        inline constexpr bool providesMetadataAtRT = false;

        template<typename T>
        inline constexpr bool
            providesMetadataAtRT<T, std::enable_if_t<providesMetadata<T> && !providesMetadataAtCT<T>>> = true;

        namespace detail
        {
            // This one gets a template argument, so that the static_assert in to_json can print the type of TObject.
            template<typename TObject>
            struct ReturnTypeFromDefault
            {
            };

            template<typename>
            inline constexpr bool False = false;

            template<typename T>
            void to_json(nlohmann::json&, ReturnTypeFromDefault<T> const&)
            {
                static_assert(
                    False<T>,
                    "You're missing metadata for a type supposed to provide some. There are three alternatives for "
                    "you: Specialise GetMetadata<YourType>, add a .metadata() method to your type, or use "
                    "AllowMissingMetadata<YourType> during the registration. For more infos, see "
                    "docs/source/usage/metadata.rst.");
            }
        } // namespace detail

        template<typename TObject, typename = void>
        struct GetMetadata
        {
            detail::ReturnTypeFromDefault<TObject> description() const
            {
                return {};
            }
        };

        // doc-include-start: GetMetdata trait
        template<typename TObject>
        struct GetMetadata<TObject, std::enable_if_t<providesMetadataAtRT<TObject>>>
        {
            // Holds a constant reference to the RT instance it's supposed to report about.
            // Omit this for the CT specialisation!
            TObject const& obj;

            nlohmann::json description() const
            {
                return obj.metadata();
            }
        };

        template<typename TObject>
        struct GetMetadata<TObject, std::enable_if_t<providesMetadataAtCT<TObject>>>
        {
            // CT version has no members. Apart from that, the interface is identical to the RT version.

            nlohmann::json description() const
            {
                return TObject::metadata();
            }
        };
        // doc-include-end: GetMetdata trait

        template<typename TObject>
        struct AllowMissingMetadata
        {
            using type = TObject;
        };

        template<typename TObject>
        struct GetMetadata<AllowMissingMetadata<TObject>> : GetMetadata<TObject>
        {
            nlohmann::json description() const
            {
                return handle(GetMetadata<TObject>::description());
            }

            static nlohmann::json handle(nlohmann::json const& result)
            {
                return result;
            }

            static nlohmann::json handle(detail::ReturnTypeFromDefault<TObject> const& result)
            {
                return nlohmann::json::object();
            }
        };

        template<typename Profiles>
        struct IncidentFieldPolicy
        {
        };

        namespace detail
        {
            std::list<std::string> boundaryNames = {"XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax"};

            template<template<typename...> typename T_Pack, typename... Profiles>
            nlohmann::json gatherMetadata(T_Pack<Profiles...>)
            {
                std::vector<nlohmann::json> collection;
                (collection.push_back(GetMetadata<Profiles>{}.description()), ...);
                return std::transform_reduce(
                    cbegin(collection),
                    cend(collection),
                    cbegin(boundaryNames),
                    nlohmann::json::object(),
                    // take by value because we will return it, so it must be owned by us:
                    [](auto final_obj, auto const& new_content)
                    {
                        final_obj.merge_patch(new_content);
                        return final_obj;
                    },
                    [](auto const& metadata, auto const& name)
                    {
                        auto result = nlohmann::json::object();
                        result[name] = metadata;
                        return result;
                    });
            }
        } // namespace detail


        template<typename Profiles>
        struct GetMetadata<IncidentFieldPolicy<Profiles>>
        {
            nlohmann::json description() const
            {
                auto gathered = detail::gatherMetadata(Profiles{});
                auto result = nlohmann::json::object();
                result["incidentField"] = gathered;
                return result;
            }
        };

    } // namespace traits
} // namespace picongpu

namespace pmacc::math
{
    template<typename T_Type, uint32_t T_dim, typename T_Navigator, typename T_Storage>
    void to_json(nlohmann::json& j, Vector<T_Type, T_dim, T_Navigator, T_Storage> const& vec)
    {
        std::vector<T_Type> stdvec{};
        for(size_t i = 0; i < T_dim; ++i)
        {
            stdvec.push_back(vec[i]);
        }
        j = stdvec;
    }
} // namespace pmacc::math
