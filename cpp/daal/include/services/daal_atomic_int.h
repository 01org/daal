/* file: daal_atomic_int.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

/*
//++
//  Declaration of class for atomic operations with int
//--
*/

#ifndef __DAAL_ATOMIC_INT_H__
#define __DAAL_ATOMIC_INT_H__

#if defined(_WIN32) || defined(_WIN64)
    #include <intrin.h>
#endif

#include "services/daal_defines.h"

namespace daal
{
namespace services
{
namespace interface1
{
/**
 * @ingroup memory
 * @{
 */

#if defined(_WIN32)
/**
 * <a name="DAAL-CLASS-SERVICES__ATOMIC"></a>
 * \brief Class that represents an atomic object
 *
 * \tparam dataType Data type of the atomic object
 */
template <typename dataType>
class DAAL_EXPORT Atomic
{
public:
    /**
     * Returns an increment of atomic object
     * \return An increment of atomic object
     */
    DAAL_FORCEINLINE dataType inc()
    {
        DAAL_ASSERT(sizeof(my_storage) == sizeof(long))
        return (dataType)(_InterlockedExchangeAdd((long *)(&my_storage), 1) + 1);
    }

    /**
     * Returns a decrement of atomic object
     * \return An decrement of atomic object
     */
    DAAL_FORCEINLINE dataType dec()
    {
        DAAL_ASSERT(sizeof(my_storage) == sizeof(long))
        return (dataType)(_InterlockedExchangeAdd((long *)(&my_storage), -1) - 1);
    }

    /**
     * Returns a result of comparing and exchanging  of atomic object
     * \param[in] expected    The value to compare with
     * \param[in] expected    The value to store after succesfull comparison
     * \return Result of comparing and exchanging
     */
    DAAL_FORCEINLINE bool compare_exchange(dataType & expected, dataType desired)
    {
        DAAL_ASSERT(sizeof(my_storage) == sizeof(long))
        auto res_val = _InterlockedCompareExchange((long volatile *)(&my_storage), desired, expected);
        _ReadWriteBarrier();
        bool result = res_val == desired ? true : false;
        return result;
    }

    /**
     * Assigns the value to atomic object
     * \param[in] value    The value to be assigned
     */
    DAAL_FORCEINLINE void set(dataType value)
    {
        _ReadWriteBarrier();
        my_storage = value;
    }

    /**
     * Returns the value of the atomic object
     * \return The value of the atomic object
     */
    DAAL_FORCEINLINE dataType get() const
    {
        dataType to_return = my_storage;
        _ReadWriteBarrier();
        return to_return;
    }

    /**
     * Constructs an atomic object
     */
    Atomic() : my_storage(0) {}

    /**
     * Constructs an atomic object from a value
     * \param[in] value The value to be assigned to the atomic object
     */
    Atomic(dataType value) : my_storage(value) {}

protected:
    dataType my_storage;

private:
    Atomic(const Atomic &);
    Atomic & operator=(const Atomic &);
};

    #if defined(_WIN64)
template <>
class DAAL_EXPORT Atomic<size_t>
{
public:
    /**
     * Returns an increment of atomic object
     * \return An increment of atomic object
     */
    DAAL_FORCEINLINE size_t inc()
    {
        DAAL_ASSERT(sizeof(my_storage) == sizeof(size_t))
        return (size_t)(_InterlockedExchangeAdd64((__int64 *)(&my_storage), 1) + 1);
    }

    /**
     * Returns a decrement of atomic object
     * \return An decrement of atomic object
     */
    DAAL_FORCEINLINE size_t dec()
    {
        DAAL_ASSERT(sizeof(my_storage) == sizeof(size_t))
        return (size_t)(_InterlockedExchangeAdd64((__int64 *)(&my_storage), -1) - 1);
    }

    /**
     * Returns a result of comparing and exchanging  of atomic object
     * \param[in] expected    The value to compare with
     * \param[in] expected    The value to store after succesfull comparison
     * \return Result of comparing and exchanging
     */
    DAAL_FORCEINLINE bool compare_exchange(size_t & expected, size_t desired)
    {
        DAAL_ASSERT(sizeof(my_storage) == sizeof(size_t))
        auto res_val = _InterlockedCompareExchange64((__int64 volatile *)(&my_storage), desired, expected);
        _ReadWriteBarrier();
        bool result = res_val == desired ? true : false;
        return result;
    }

    /**
     * Assigns the value to atomic object
     * \param[in] value    The value to be assigned
     */
    DAAL_FORCEINLINE void set(size_t value)
    {
        _ReadWriteBarrier();
        my_storage = value;
    }

    /**
     * Returns the value of the atomic object
     * \return The value of the atomic object
     */
    DAAL_FORCEINLINE size_t get() const
    {
        size_t to_return = my_storage;
        _ReadWriteBarrier();
        return to_return;
    }

    /**
     * Constructs an atomic object
     */
    Atomic() : my_storage(0) {}

    /**
     * Constructs an atomic object from a value
     * \param[in] value The value to be assigned to the atomic object
     */
    Atomic(size_t value) : my_storage(value) {}

protected:
    size_t my_storage;

private:
    Atomic(const Atomic &);
    Atomic & operator=(const Atomic &);
};
    #endif // _WIN64
#endif     // _WIN32

#if !defined(_WIN32)
/**
 * <a name="DAAL-CLASS-SERVICES__ATOMIC"></a>
 * \brief Class that represents an atomic object
 *
 * \tparam dataType Data type of the atomic object
 */
template <typename dataType>
class DAAL_EXPORT Atomic
{
public:
    /**
     * Returns an increment of atomic object
     * \return An increment of atomic object
     */
    DAAL_FORCEINLINE dataType inc() { return __atomic_add_fetch(&my_storage, 1, __ATOMIC_SEQ_CST); }

    /**
     * Returns a decrement of atomic object
     * \return An decrement of atomic object
     */
    DAAL_FORCEINLINE dataType dec() { return __atomic_sub_fetch(&my_storage, 1, __ATOMIC_SEQ_CST); }

    /**
     * Returns a result of comparing and exchanging  of atomic object
     * \param[in] expected    The value to compare with
     * \param[in] expected    The value to store after succesfull comparison
     * \return Result of comparing and exchanging
     */
    DAAL_FORCEINLINE bool compare_exchange(dataType & expected, dataType desired)
    {
        return __atomic_compare_exchange(&my_storage, &expected, &desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }

    /**
     * Assigns the value to atomic object
     * \param[in] value    The value to be assigned
     */
    DAAL_FORCEINLINE void set(dataType value)
    {
        __asm__ __volatile__("" : : : "memory");
        __atomic_store_n(&my_storage, value, __ATOMIC_RELEASE);
    }

    /**
     * Returns the value of the atomic object
     * \return The value of the atomic object
     */
    DAAL_FORCEINLINE dataType get() const
    {
        dataType to_return = __atomic_load_n(&my_storage, __ATOMIC_ACQUIRE);
        __asm__ __volatile__("" : : : "memory");
        return to_return;
    }

    /**
     * Constructs an atomic object
     */
    Atomic() : my_storage(0) {}

    /**
     * Constructs an atomic object from a value
     * \param[in] value The value to be assigned to the atomic object
     */
    Atomic(dataType value) : my_storage(value) {}

protected:
    dataType my_storage;

private:
    Atomic(const Atomic &);
    Atomic & operator=(const Atomic &);
};
#endif // !_WIN32

/** @} */

} // namespace interface1

using interface1::Atomic;

typedef Atomic<int> AtomicInt;
typedef Atomic<size_t> AtomicSizeT;

} // namespace services
} // namespace daal

#endif
