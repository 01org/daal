/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <algorithm>
#include "oneapi/dal/detail/hash_map.hpp"

#define __ONEDAL_REGISTER_SERIALIZABLE__VAR2__(name, unique) name##unique
#define __ONEDAL_REGISTER_SERIALIZABLE__VAR__(unique) \
    __ONEDAL_REGISTER_SERIALIZABLE__VAR2__(__serializable_stub, unique)
#define __ONEDAL_REGISTER_SERIALIZABLE__(T)                                                   \
    namespace {                                                                               \
    [[maybe_unused]] volatile static bool __ONEDAL_REGISTER_SERIALIZABLE__VAR__(__LINE__) =   \
        ::oneapi::dal::detail::serializable_registry::instance().register_default_factory<T>( \
            T::serialization_id());                                                           \
    }

namespace oneapi::dal::detail {

struct serialization_accessor {
    serialization_accessor() = delete;

    template <typename Object, typename... Args>
    static auto serialize(const Object& object, Args&&... args) {
        return object.serialize(std::forward<Args>(args)...);
    }

    template <typename Object, typename... Args>
    static auto deserialize(Object& object, Args&&... args) {
        return object.deserialize(std::forward<Args>(args)...);
    }
};

/// Archive interface for deserialization
class input_archive_iface {
public:
    virtual ~input_archive_iface() = default;
    virtual void prologue() = 0;
    virtual void epilogue() = 0;
    virtual void deserialize(void* data, data_type dtype) = 0;
    virtual void deserialize(void* data, data_type dtype, std::int64_t count) = 0;
};

/// Archive interface for serialization
class output_archive_iface {
public:
    virtual ~output_archive_iface() = default;
    virtual void prologue() = 0;
    virtual void epilogue() = 0;
    virtual void serialize(const void* data, data_type dtype) = 0;
    virtual void serialize(const void* data, data_type dtype, std::int64_t count) = 0;
};

template <typename T>
inline constexpr bool is_trivially_serializable_v = std::is_arithmetic_v<T> || std::is_enum_v<T>;

template <typename T>
using enable_if_trivially_serializable_t = std::enable_if_t<is_trivially_serializable_v<T>>;

template <typename T>
using enable_if_user_serializable_t = std::enable_if_t<!is_trivially_serializable_v<T>>;

template <typename T,
          bool is_enum = std::is_enum_v<T>,
          typename = enable_if_trivially_serializable_t<T>>
struct trivial_serialization_type;

template <typename T>
struct trivial_serialization_type<T, false> {
    using type = T;
};

template <>
struct trivial_serialization_type<bool, false> {
    using type = std::uint8_t;
};

template <typename T>
struct trivial_serialization_type<T, true> {
    using type = std::underlying_type_t<T>;
};

template <typename T>
using trivial_serialization_type_t = typename trivial_serialization_type<T>::type;

template <typename Archive>
class input_archive_impl : public base, public input_archive_iface {
public:
    explicit input_archive_impl(Archive& archive) : archive_(archive) {}

    void prologue() override {
        archive_.prologue();
    }

    void epilogue() override {
        archive_.epilogue();
    }

    void deserialize(void* data, data_type dtype) override {
        archive_(data, dtype);
    }

    void deserialize(void* data, data_type dtype, std::int64_t count) override {
        archive_(data, dtype, count);
    }

private:
    std::remove_reference_t<Archive>& archive_;
};

template <typename Archive>
class output_archive_impl : public base, public output_archive_iface {
public:
    explicit output_archive_impl(Archive& archive) : archive_(archive) {}

    void prologue() override {
        archive_.prologue();
    }

    void epilogue() override {
        archive_.epilogue();
    }

    void serialize(const void* data, data_type dtype) override {
        archive_(data, dtype);
    }

    void serialize(const void* data, data_type dtype, std::int64_t count) override {
        archive_(data, dtype, count);
    }

private:
    std::remove_reference_t<Archive>& archive_;
};

template <typename Iface>
class archive_base : public base {
protected:
    explicit archive_base(Iface* impl) noexcept : impl_(impl) {
        ONEDAL_ASSERT(impl);
    }

    template <typename DerivedIface = Iface>
    DerivedIface& get_impl() {
        return static_cast<DerivedIface&>(*impl_);
    }

    template <typename DerivedIface = Iface>
    const DerivedIface& get_impl() const {
        return static_cast<const DerivedIface&>(*impl_);
    }

private:
    pimpl<Iface> impl_;
};

class input_archive : public archive_base<input_archive_iface> {
    using base_t = archive_base<input_archive_iface>;

public:
    template <typename Archive>
    explicit input_archive(Archive& archive) : base_t(new input_archive_impl<Archive>{ archive }) {}

    void prologue() {
        get_impl().prologue();
    }

    void epilogue() {
        get_impl().epilogue();
    }

    template <typename... Args>
    void operator()(Args&&... args) {
        (process(std::forward<Args>(args)), ...);
    }

    template <typename T>
    void range(T* begin, T* end) {
        ONEDAL_ASSERT(begin);
        ONEDAL_ASSERT(end);
        ONEDAL_ASSERT(begin <= end);
        process(begin, end);
    }

    template <typename T>
    T pop() {
        T value{};
        (*this)(value);
        return value;
    }

private:
    template <typename T, enable_if_trivially_serializable_t<T>* = nullptr>
    void process(T& value) {
        using trivial_t = trivial_serialization_type_t<T>;
        trivial_t value_to_deserialize;
        get_impl().deserialize(&value_to_deserialize, make_data_type<trivial_t>());
        value = T(value_to_deserialize);
    }

    template <typename T, enable_if_user_serializable_t<T>* = nullptr>
    void process(T& value) {
        serialization_accessor::deserialize(value, *this);
    }

    template <typename T>
    void process(T* begin, T* end) {
        std::int64_t count = end - begin;
        get_impl().deserialize(begin, make_data_type<T>(), count);
    }
};

class output_archive : public archive_base<output_archive_iface> {
    using base_t = archive_base<output_archive_iface>;

public:
    template <typename Archive>
    explicit output_archive(Archive& archive)
            : base_t(new output_archive_impl<Archive>{ archive }) {}

    void prologue() {
        get_impl().prologue();
    }

    void epilogue() {
        get_impl().epilogue();
    }

    template <typename... Args>
    void operator()(Args&&... args) {
        (process(std::forward<Args>(args)), ...);
    }

    template <typename T>
    void range(const T* begin, const T* end) {
        ONEDAL_ASSERT(begin);
        ONEDAL_ASSERT(end);
        ONEDAL_ASSERT(begin <= end);
        process(begin, end);
    }

private:
    template <typename T, enable_if_trivially_serializable_t<T>* = nullptr>
    void process(const T& value) {
        using trivial_t = trivial_serialization_type_t<T>;
        const auto value_to_serialize = trivial_t(value);
        get_impl().serialize(&value_to_serialize, make_data_type<trivial_t>());
    }

    template <typename T, enable_if_user_serializable_t<T>* = nullptr>
    void process(const T& value) {
        serialization_accessor::serialize(value, *this);
    }

    template <typename T>
    void process(const T* begin, const T* end) {
        const std::int64_t count = end - begin;
        get_impl().serialize(begin, make_data_type<T>(), count);
    }
};

/// Interface that each serializable class must implement to support polymorphic serialization
class serializable_iface {
public:
    virtual ~serializable_iface() = default;
    virtual std::uint64_t get_serialization_id() const = 0;
    virtual void serialize(detail::output_archive& ar) const = 0;
    virtual void deserialize(detail::input_archive& ar) = 0;
};

class serializable_factory_iface {
public:
    virtual ~serializable_factory_iface() = default;
    virtual serializable_iface* make() const = 0;
};

template <typename T>
class default_serializable_factory : public serializable_factory_iface {
public:
    static const default_serializable_factory& instance() {
        static default_serializable_factory factory;
        return factory;
    }

    T* make() const override {
        return new T{};
    }

private:
    default_serializable_factory() = default;
};

class serializable_registry : public base {
public:
    static serializable_registry& instance();

    template <typename T>
    T* make(std::uint64_t serialization_id) {
        ONEDAL_ASSERT(factories_.has(serialization_id),
                      "Factory with requested serialization_id was not registered");

        auto serializable = factories_.get(serialization_id)->make();
        ONEDAL_ASSERT(serializable, "Factory produced null object");

        if (serializable->get_serialization_id() != serialization_id) {
            throw invalid_argument{ error_messages::archive_content_does_not_match_type() };
        }

        auto object = dynamic_cast<T*>(serializable);
        if (!object) {
            throw invalid_argument{ error_messages::archive_content_does_not_match_type() };
        }

        return object;
    }

    template <typename T>
    bool register_default_factory(std::uint64_t serialization_id) {
        return register_factory(serialization_id, &default_serializable_factory<T>::instance());
    }

private:
    serializable_registry() = default;

    bool register_factory(std::uint64_t serialization_id,
                          const serializable_factory_iface* factory) {
        ONEDAL_ASSERT(factory);
        ONEDAL_ASSERT(!factories_.has(serialization_id),
                      "Factory with the provided serialization_id is already registered");

        factories_.set(serialization_id, factory);
        return true;
    }

    static constexpr std::int64_t hash_map_capacity = 1024;
    hash_map<std::uint64_t, const serializable_factory_iface*> factories_{ hash_map_capacity };
};

template <std::uint64_t SerializationId>
class serializable : public base, public serializable_iface {
public:
    static std::uint64_t serialization_id() {
        return SerializationId;
    }

    std::uint64_t get_serialization_id() const override {
        return serialization_id();
    }
};

template <typename T>
inline serializable_iface& get_serializable(T* object) {
    auto serializable = dynamic_cast<serializable_iface*>(object);
    if (!serializable) {
        throw invalid_argument{ error_messages::object_is_not_serializable() };
    }
    return *serializable;
}

template <typename T>
inline serializable_iface& get_serializable(const std::shared_ptr<T>& object) {
    return get_serializable(object.get());
}

template <typename T>
inline void serialize_polymorphic(T* serializable_object, output_archive& archive) {
    const bool is_not_nullptr = (serializable_object != nullptr);
    archive(is_not_nullptr);

    if (is_not_nullptr) {
        auto& serializable = get_serializable(serializable_object);
        archive(serializable.get_serialization_id());
        serializable.serialize(archive);
    }
}

template <typename T>
inline void serialize_polymorphic_shared(const std::shared_ptr<T>& serializable_object,
                                         output_archive& archive) {
    serialize_polymorphic(serializable_object.get(), archive);
}

template <typename T>
inline T* deserialize_polymorphic(input_archive& archive,
                                  const std::initializer_list<std::uint64_t>& allowed_ids = {}) {
    const bool is_not_nullptr = archive.pop<bool>();
    if (!is_not_nullptr) {
        return nullptr;
    }

    const auto serialization_id = archive.pop<std::uint64_t>();
    if (allowed_ids.size() > 0) {
        const auto it = std::find(allowed_ids.begin(), allowed_ids.end(), serialization_id);
        if (it == allowed_ids.end()) {
            throw invalid_argument{ error_messages::archive_content_does_not_match_type() };
        }
    }

    T* object_ptr = serializable_registry::instance().make<T>(serialization_id);
    get_serializable(object_ptr).deserialize(archive);
    return object_ptr;
}

template <typename T>
inline std::shared_ptr<T> deserialize_polymorphic_shared(
    input_archive& archive,
    const std::initializer_list<std::uint64_t>& allowed_ids = {}) {
    return std::shared_ptr<T>{ deserialize_polymorphic<T>(archive, allowed_ids) };
}

template <typename T>
inline void deserialize_polymorphic_shared(
    std::shared_ptr<T>& serializable_object,
    input_archive& archive,
    const std::initializer_list<std::uint64_t>& allowed_ids = {}) {
    serializable_object.reset(deserialize_polymorphic<T>(archive, allowed_ids));
}

template <typename T, typename OutputArchive>
inline void serialize(const T& value, OutputArchive& archive) {
    output_archive internal_archive{ archive };
    internal_archive.prologue();
    internal_archive(value);
    internal_archive.epilogue();
}

template <typename T, typename InputArchive>
inline void deserialize(T& value, InputArchive& archive) {
    input_archive internal_archive{ archive };
    internal_archive.prologue();
    internal_archive(value);
    internal_archive.epilogue();
}

} // namespace oneapi::dal::detail
