//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___BIT_ROTATE_H
#define _LIBCPP___CXX03___BIT_ROTATE_H

#include <__cxx03/__config>
#include <__cxx03/__type_traits/is_unsigned_integer.h>
#include <__cxx03/limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// Writing two full functions for rotl and rotr makes it easier for the compiler
// to optimize the code. On x86 this function becomes the ROL instruction and
// the rotr function becomes the ROR instruction.
template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _Tp __rotl(_Tp __x, int __s) _NOEXCEPT {
  static_assert(__libcpp_is_unsigned_integer<_Tp>::value, "__rotl requires an unsigned integer type");
  const int __N = numeric_limits<_Tp>::digits;
  int __r       = __s % __N;

  if (__r == 0)
    return __x;

  if (__r > 0)
    return (__x << __r) | (__x >> (__N - __r));

  return (__x >> -__r) | (__x << (__N + __r));
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _Tp __rotr(_Tp __x, int __s) _NOEXCEPT {
  static_assert(__libcpp_is_unsigned_integer<_Tp>::value, "__rotr requires an unsigned integer type");
  const int __N = numeric_limits<_Tp>::digits;
  int __r       = __s % __N;

  if (__r == 0)
    return __x;

  if (__r > 0)
    return (__x >> __r) | (__x << (__N - __r));

  return (__x << -__r) | (__x >> (__N + __r));
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___BIT_ROTATE_H
