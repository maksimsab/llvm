///===- llvm/Frontend/Offloading/PropertySet.h ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
///===---------------------------------------------------------------------===//
/// \file This file defines PropertySetRegistry and PropertyValue types and
/// provides helper functions to translate PropertySetRegistry from/to JSON.
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OFFLOADING_PROPERTY_SET_H
#define LLVM_FRONTEND_OFFLOADING_PROPERTY_SET_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

#include <map>
#include <variant>

namespace llvm {
class raw_ostream;
class MemoryBufferRef;

namespace offloading {

using ByteArray = SmallVector<char, 0>;

constexpr int PROPERTY_VALUE_INTEGER_TYPE_INDEX = 0;
constexpr int PROPERT_VALUE_BYTE_ARRAY_INDEX = 1;
using PropertyValue = std::variant<uint32_t, ByteArray>;
using PropertySet = std::map<std::string, PropertyValue>;
using PropertySetRegistry = std::map<std::string, PropertySet>;

LLVM_ABI void writePropertiesToJSON(const PropertySetRegistry &P,
                                    raw_ostream &O);
LLVM_ABI Expected<PropertySetRegistry>
readPropertiesFromJSON(MemoryBufferRef Buf);

} // namespace offloading
} // namespace llvm

#endif // LLVM_FRONTEND_OFFLOADING_PROPERTY_SET_H
